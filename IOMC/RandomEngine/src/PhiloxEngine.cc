#include "IOMC/RandomEngine/interface/PhiloxEngine.h"

#include "CLHEP/Random/engineIDulong.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>

namespace {
  constexpr char const* const kBeginTag = "PhiloxEngine-begin";
  constexpr char const* const kEndTag = "PhiloxEngine-end";
}  // namespace

namespace edm {

  PhiloxEngine::PhiloxEngine() : PhiloxEngine(0) {}

  PhiloxEngine::PhiloxEngine(long seed) : id_{}, stream_(id_) { setSeed(seed, 0); }

  double PhiloxEngine::flat() {
    std::uint32_t const low = nextWord();
    return edm::philox::openUniformDouble(low, nextWord());
  }

  void PhiloxEngine::flatArray(int const size, double* vect) {
    for (int i = 0; i < size; ++i) {
      vect[i] = flat();
    }
  }

  PhiloxEngine::operator float() { return edm::philox::openUniformFloat(nextWord()); }

  void PhiloxEngine::setSeed(long seed, int) {
    theSeed = seed;
    auto const bits = static_cast<std::uint64_t>(seed);
    selectStream(
        edm::philox::Counter{{static_cast<std::uint32_t>(bits), static_cast<std::uint32_t>(bits >> 32), 0, 0}});
  }

  void PhiloxEngine::setSeeds(long const* seeds, int seedNum) {
    edm::philox::Counter id{};
    unsigned int const maxWords = seedNum < 1 ? kWordsPerBlock : std::min<unsigned int>(seedNum, kWordsPerBlock);
    for (unsigned int i = 0; i < maxWords; ++i) {
      if (seedNum < 1 and seeds[i] == 0) {
        break;
      }
      id[i] = static_cast<std::uint32_t>(seeds[i]);
    }
    theSeed = seeds[0];
    selectStream(id);
  }

  void PhiloxEngine::selectStream(edm::philox::Counter const& id) {
    id_ = id;
    stream_ = edm::philox::EventStream(id_);
    nextBlock_ = 0;
    position_ = kWordsPerBlock;
  }

  edm::philox::Counter PhiloxEngine::blockBits(std::uint64_t block) const {
    return stream_.bits(static_cast<std::uint32_t>(block >> 32), static_cast<std::uint32_t>(block));
  }

  std::uint32_t PhiloxEngine::nextWord() {
    if (position_ == kWordsPerBlock) {
      buffer_ = blockBits(nextBlock_);
      ++nextBlock_;
      position_ = 0;
    }
    return buffer_[position_++];
  }

  std::vector<unsigned long> PhiloxEngine::put() const {
    return {CLHEP::engineIDulong<PhiloxEngine>(),
            id_[0],
            id_[1],
            id_[2],
            id_[3],
            static_cast<std::uint32_t>(nextBlock_),
            static_cast<std::uint32_t>(nextBlock_ >> 32),
            position_};
  }

  bool PhiloxEngine::get(std::vector<unsigned long> const& v) {
    if (v.size() != kStateSize or v[0] != CLHEP::engineIDulong<PhiloxEngine>()) {
      return false;
    }
    return getState(v);
  }

  bool PhiloxEngine::getState(std::vector<unsigned long> const& v) {
    if (v.size() != kStateSize) {
      return false;
    }
    std::uint64_t const nextBlock = (static_cast<std::uint64_t>(v[6]) << 32) | static_cast<std::uint32_t>(v[5]);
    // A partly used block needs a previous block. The state does not change for an invalid input.
    if (v[7] > kWordsPerBlock or (v[7] < kWordsPerBlock and nextBlock == 0)) {
      return false;
    }
    selectStream(edm::philox::Counter{{static_cast<std::uint32_t>(v[1]),
                                       static_cast<std::uint32_t>(v[2]),
                                       static_cast<std::uint32_t>(v[3]),
                                       static_cast<std::uint32_t>(v[4])}});
    nextBlock_ = nextBlock;
    position_ = v[7];
    if (position_ < kWordsPerBlock) {
      buffer_ = blockBits(nextBlock_ - 1);
    }
    return true;
  }

  std::ostream& PhiloxEngine::put(std::ostream& os) const {
    os << kBeginTag << "\n";
    for (auto value : put()) {
      os << value << "\n";
    }
    os << kEndTag << "\n";
    return os;
  }

  std::istream& PhiloxEngine::get(std::istream& is) {
    std::string tag;
    is >> tag;
    if (tag != kBeginTag) {
      is.clear(std::ios::badbit | is.rdstate());
      return is;
    }
    return getState(is);
  }

  std::istream& PhiloxEngine::getState(std::istream& is) {
    std::vector<unsigned long> v(kStateSize);
    for (auto& value : v) {
      is >> value;
    }
    std::string tag;
    is >> tag;
    if (not is or tag != kEndTag or not get(v)) {
      is.clear(std::ios::badbit | is.rdstate());
    }
    return is;
  }

  void PhiloxEngine::saveStatus(char const filename[]) const {
    std::ofstream file(filename);
    put(file);
  }

  void PhiloxEngine::restoreStatus(char const filename[]) {
    std::ifstream file(filename);
    if (not get(file)) {
      throw Exception(errors::Configuration)
          << "PhiloxEngine::restoreStatus could not read the state from file \"" << filename << "\"\n";
    }
  }

  void PhiloxEngine::showStatus() const {
    std::cout << "PhiloxEngine stream " << std::hex << id_[0] << " " << id_[1] << " " << id_[2] << " " << id_[3]
              << std::dec << " next block " << nextBlock_ << " position " << position_ << "\n";
  }

}  // namespace edm

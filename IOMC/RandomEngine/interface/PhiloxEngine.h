#ifndef IOMC_RandomEngine_PhiloxEngine_h
#define IOMC_RandomEngine_PhiloxEngine_h

// A CLHEP engine on the counter-based Philox4x32-10 generator.
// setSeeds with four 32-bit words selects the stream; the engine then returns the
// blocks of that stream in order. A reseed has a constant, small cost and no warm-up.

#include "CLHEP/Random/RandomEngine.h"
#include "FWCore/Utilities/interface/Philox.h"

#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace edm {

  class PhiloxEngine final : public CLHEP::HepRandomEngine {
  public:
    PhiloxEngine();
    explicit PhiloxEngine(long seed);

    // Uniform in (0, 1) with 52 random bits.
    double flat() override;
    void flatArray(int const size, double* vect) override;

    // The stream {low 32 bits of seed, high 32 bits of seed, 0, 0}.
    void setSeed(long seed, int) override;
    // Up to four 32-bit words select the stream. With seedNum < 1 the list ends at the first 0.
    void setSeeds(long const* seeds, int seedNum) override;

    void saveStatus(char const filename[] = "PhiloxEngine.conf") const override;
    void restoreStatus(char const filename[] = "PhiloxEngine.conf") override;
    void showStatus() const override;

    std::string name() const override { return engineName(); }
    static std::string engineName() { return "PhiloxEngine"; }

    std::ostream& put(std::ostream& os) const override;
    std::istream& get(std::istream& is) override;
    std::istream& getState(std::istream& is) override;

    std::vector<unsigned long> put() const override;
    bool get(std::vector<unsigned long> const& v) override;
    bool getState(std::vector<unsigned long> const& v) override;

    operator double() override { return flat(); }
    // Uniform in (0, 1) with 23 random bits.
    operator float() override;
    operator unsigned int() override { return nextWord(); }

  private:
    static constexpr unsigned int kWordsPerBlock = 4;
    // The engine id, the 4 words of the stream id, the low and the high word of nextBlock_, position_.
    static constexpr std::size_t kStateSize = 8;

    std::uint32_t nextWord();
    void selectStream(edm::philox::Counter const& id);
    edm::philox::Counter blockBits(std::uint64_t block) const;

    edm::philox::Counter id_;
    edm::philox::EventStream stream_;
    // Index of the next block to compute; the buffer holds block nextBlock_ - 1.
    std::uint64_t nextBlock_ = 0;
    edm::philox::Counter buffer_{};
    unsigned int position_ = kWordsPerBlock;
  };

}  // namespace edm

#endif

#ifndef DataFormats_L1TParticleFlow_jets_h
#define DataFormats_L1TParticleFlow_jets_h

#include "DataFormats/L1TParticleFlow/interface/datatypes.h"
#include "DataFormats/L1TParticleFlow/interface/gt_datatypes.h"
#include "DataFormats/L1TParticleFlow/interface/bit_encoding.h"
#include <array>
#include <algorithm>
#include <cstdint>
#include <vector>
#include <unordered_map>

namespace l1ct {

  // all possible tag categories (can be extended for new / separate taggers)
  class JetTagClass {
  public:
    enum JetTagClassValue : uint8_t { b, c, uds, g, tau_p, tau_n, mu, e };
    JetTagClass() = default;
    JetTagClass(JetTagClassValue aJetTagClassValue) : value_(aJetTagClassValue) {}
    JetTagClass(std::string aJetTagClassValueString) {
      auto it = labels_.find(aJetTagClassValueString);
      if (it != labels_.end()) {
        value_ = it->second;
      } else {
        // TODO throw an error
        value_ = JetTagClass::JetTagClassValue::uds;
      }
    }

    inline bool operator==(const JetTagClass &other) const { return value_ == other.value_; }

  private:
    JetTagClassValue value_;
    static const std::unordered_map<std::string, JetTagClassValue> labels_;

    friend std::ostream &operator<<(std::ostream &ost, const l1ct::JetTagClass &jtc) {
      auto it = std::find_if(
          std::begin(jtc.labels_), std::end(jtc.labels_), [&jtc](auto &&p) { return p.second == jtc.value_; });
      if (it != std::end(jtc.labels_)) {
        ost << it->first;
      }
      return ost;
    }

  };  // JetTagClass

  // Define a separate class/struct for jet tag handling
  struct JetTagClassHandler {
    static const unsigned NTagFields = 8;
    static const JetTagClass tagClassesDefault_[NTagFields];

    JetTagClass tagClassesArray[NTagFields];

    JetTagClassHandler() {
      // Copy the default values to the array
      for (unsigned i = 0; i < NTagFields; i++) {
        tagClassesArray[i] = tagClassesDefault_[i];
      }
    }
  };

  struct Jet {
    pt_t hwPt;
    glbeta_t hwEta;
    glbphi_t hwPhi;
    z0_t hwZ0;

    static const unsigned NTagFields = 8;
    jet_tag_score_t hwTagScores[NTagFields];

    inline bool operator==(const Jet &other) const {
      bool eq = hwPt == other.hwPt && hwEta == other.hwEta && hwPhi == other.hwPhi && hwZ0 == other.hwZ0;
      for (unsigned i = 0; i < NTagFields; i++) {
        eq = eq && hwTagScores[i] == other.hwTagScores[i];
      }
      return eq;
    }

    inline bool operator>(const Jet &other) const { return hwPt > other.hwPt; }
    inline bool operator<(const Jet &other) const { return hwPt < other.hwPt; }

    inline void clear() {
      hwPt = 0;
      hwEta = 0;
      hwPhi = 0;
      hwZ0 = 0;
      for (unsigned i = 0; i < NTagFields; i++) {
        hwTagScores[i] = 0;
      }
    }

    int intPt() const { return Scales::intPt(hwPt); }
    int intEta() const { return hwEta.to_int(); }
    int intPhi() const { return hwPhi.to_int(); }
    float floatPt() const { return Scales::floatPt(hwPt); }
    float floatEta() const { return Scales::floatEta(hwEta); }
    float floatPhi() const { return Scales::floatPhi(hwPhi); }
    float floatZ0() const { return Scales::floatZ0(hwZ0); }
    std::vector<float> floatIDScores() const {
      std::vector<float> scores(NTagFields);
      for (unsigned i = 0; i < NTagFields; i++) {
        scores[i] = (float)hwTagScores[i];
      }
      return scores;
    }

    static const int BITWIDTH =
        pt_t::width + glbeta_t::width + glbphi_t::width + z0_t::width + NTagFields * id_score_t::width;
    inline ap_uint<BITWIDTH> pack_ap() const {
      ap_uint<BITWIDTH> ret;
      unsigned int start = 0;
      pack_into_bits(ret, start, hwPt);
      pack_into_bits(ret, start, hwEta);
      pack_into_bits(ret, start, hwPhi);
      pack_into_bits(ret, start, hwZ0);
      for (unsigned i = 0; i < NTagFields; i++) {
        pack_into_bits(ret, start, hwTagScores[i]);
      }
      return ret;
    }

    inline std::array<uint64_t, 2> pack() const {
      std::array<uint64_t, 2> packed = {{0, 0}};
      ap_uint<BITWIDTH> bits = this->pack_ap();
      packed[0] = bits(63, 0);
      packed[1] = bits(BITWIDTH - 1, 64);  // for when there are more than 64 bits in the word
      return packed;
    }

    inline static Jet unpack_ap(const ap_uint<BITWIDTH> &src) {
      Jet ret;
      ret.initFromBits(src);
      return ret;
    }

    inline void initFromBits(const ap_uint<BITWIDTH> &src) {
      unsigned int start = 0;
      unpack_from_bits(src, start, hwPt);
      unpack_from_bits(src, start, hwEta);
      unpack_from_bits(src, start, hwPhi);
      unpack_from_bits(src, start, hwZ0);
      for (unsigned i = 0; i < NTagFields; i++) {
        unpack_from_bits(src, start, hwTagScores[i]);
      }
    }

    inline static Jet unpack(const std::array<uint64_t, 2> &src) {
      // just one set while the word has fewer than 64 bits
      ap_uint<BITWIDTH> bits;
      bits(63, 0) = src[0];
      bits(BITWIDTH - 1, 64) = src[1];
      return unpack_ap(bits);
    }

    inline static Jet unpack(long long unsigned int &src) {
      // unpack from single 64b int
      ap_uint<BITWIDTH> bits = src;
      return unpack_ap(bits);
    }

    l1gt::Jet toGT() const {
      l1gt::Jet j;
      j.valid = hwPt != 0;
      j.v3.pt = CTtoGT_pt(hwPt);
      j.v3.phi = CTtoGT_phi(hwPhi);
      j.v3.eta = CTtoGT_eta(hwEta);
      j.z0(l1ct::z0_t::width - 1, 0) = hwZ0(l1ct::z0_t::width - 1, 0);
      for (unsigned i = 0; i < NTagFields; i++) {
        j.hwTagScores[i] = hwTagScores[i];
      }
      return j;
    }
  };

  inline void clear(Jet &c) { c.clear(); }

}  // namespace l1ct

#endif

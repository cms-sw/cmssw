#ifndef DataFormats_L1TParticleFlow_PFCluster_h
#define DataFormats_L1TParticleFlow_PFCluster_h

#include <vector>
#include <variant>
#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/L1TParticleFlow/interface/layer1_objs.h"
#include <ap_int.h>

namespace l1t {
  namespace io_v1 {
    class PFCluster : public L1Candidate {
    public:
      /// constituent information. note that this is not going to be available in the hardware!
      typedef std::pair<edm::Ptr<l1t::L1Candidate>, float> ConstituentAndFraction;
      typedef std::vector<ConstituentAndFraction> ConstituentsAndFractions;

      PFCluster() {}
      PFCluster(float pt,
                float eta,
                float phi,
                float hOverE = 0,
                bool isEM = false,
                float ptError = 0,
                int hwpt = 0,
                int hweta = 0,
                int hwphi = 0)
          : L1Candidate(PolarLorentzVector(pt, eta, phi, 0), hwpt, hweta, hwphi, /*hwQuality=*/isEM ? 1 : 0),
            hOverE_(hOverE),
            ptError_(ptError),
            encoding_(HWEncoding::None),
            digiDataW0_(0),
            digiDataW1_(0) {
        setPdgId(isEM ? 22 : 130);  // photon : non-photon(K0)
      }
      PFCluster(const LorentzVector& p4,
                float hOverE,
                bool isEM,
                float ptError = 0,
                int hwpt = 0,
                int hweta = 0,
                int hwphi = 0)
          : L1Candidate(p4, hwpt, hweta, hwphi, /*hwQuality=*/isEM ? 1 : 0), hOverE_(hOverE), ptError_(ptError) {
        setPdgId(isEM ? 22 : 130);  // photon : non-photon(K0)
      }

      enum class HWEncoding { None = 0, Had = 1, Em = 2 };

      float hOverE() const { return hOverE_; }
      void setHOverE(float hOverE) { hOverE_ = hOverE; }

      // NOTE: this might not be consistent with the value stored in the HW digi
      float emEt() const {
        if (hOverE_ == -1)
          return 0;
        return pt() / (1 + hOverE_);
      }

      // change the pt. H/E also recalculated to keep emEt constant
      void calibratePt(float newpt, float preserveEmEt = true);

      /// constituent information. note that this is not going to be available in the hardware!
      const ConstituentsAndFractions& constituentsAndFractions() const { return constituents_; }
      /// adds a candidate to this cluster; note that this only records the information, it's up to you to also set the 4-vector appropriately
      void addConstituent(const edm::Ptr<l1t::L1Candidate>& cand, float fraction = 1.0) {
        constituents_.emplace_back(cand, fraction);
      }

      float ptError() const { return ptError_; }
      void setPtError(float ptError) { ptError_ = ptError; }

      bool isEM() const { return hwQual(); }
      void setIsEM(bool isEM) { setHwQual(isEM); }
      unsigned int hwEmID() const { return hwQual(); }

      std::variant<l1ct::HadCaloObj, l1ct::EmCaloObj> caloDigiObj() const {
        switch (encoding_) {
          case HWEncoding::Had:
            return l1ct::HadCaloObj::unpack(binaryWord<l1ct::HadCaloObj::BITWIDTH>());
          case HWEncoding::Em:
            return l1ct::EmCaloObj::unpack(binaryWord<l1ct::EmCaloObj::BITWIDTH>());
          default:
            throw std::runtime_error("No encoding for this cluster");
        }
      }

      void setCaloDigi(const l1ct::HadCaloObj& obj) { setBinaryWord(obj.pack(), HWEncoding::Had); }

      void setCaloDigi(const l1ct::EmCaloObj& obj) { setBinaryWord(obj.pack(), HWEncoding::Em); }

      HWEncoding encoding() const { return encoding_; }

    private:
      float hOverE_, ptError_;

      ConstituentsAndFractions constituents_;

      template <int N>
      void setBinaryWord(ap_uint<N> word, HWEncoding encoding) {
        digiDataW0_ = word;
        digiDataW1_ = (word >> 64);
        encoding_ = encoding;
      }

      template <int N>
      ap_uint<N> binaryWord() const {
        return ap_uint<N>(digiDataW0_) | (ap_uint<N>(digiDataW1_) << 64);
      }

      HWEncoding encoding_;
      uint64_t digiDataW0_;
      uint64_t digiDataW1_;
    };
  }  // namespace io_v1
  using PFCluster = io_v1::PFCluster;
  typedef std::vector<l1t::PFCluster> PFClusterCollection;
  typedef edm::Ref<l1t::PFClusterCollection> PFClusterRef;
}  // namespace l1t
#endif

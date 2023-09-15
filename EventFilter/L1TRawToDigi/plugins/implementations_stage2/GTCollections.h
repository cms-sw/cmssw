#ifndef GTCollections_h
#define GTCollections_h

#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/Muon.h"

#include "DataFormats/L1TGlobal/interface/GlobalAlgBlk.h"
#include "DataFormats/L1TGlobal/interface/GlobalExtBlk.h"

//#include "EventFilter/L1TRawToDigi/interface/UnpackerCollections.h"
#include "L1TObjectCollections.h"

namespace l1t {
  namespace stage2 {
    class GTCollections : public L1TObjectCollections {
    public:
      GTCollections(edm::Event& e)
          : L1TObjectCollections(e), algBlk_(new GlobalAlgBlkBxCollection()), extBlk_(new GlobalExtBlkBxCollection()) {
        std::generate(muons_.begin(), muons_.end(), [] { return std::make_unique<MuonBxCollection>(); });
        std::generate(
            muonShowers_.begin(), muonShowers_.end(), [] { return std::make_unique<MuonShowerBxCollection>(); });
        std::generate(egammas_.begin(), egammas_.end(), [] { return std::make_unique<EGammaBxCollection>(); });
        std::generate(etsums_.begin(), etsums_.end(), [] { return std::make_unique<EtSumBxCollection>(); });
        std::generate(zdcsums_.begin(), zdcsums_.end(), [] { return std::make_unique<EtSumBxCollection>(); });
        std::generate(jets_.begin(), jets_.end(), [] { return std::make_unique<JetBxCollection>(); });
        std::generate(taus_.begin(), taus_.end(), [] { return std::make_unique<TauBxCollection>(); });
      };

      ~GTCollections() override;

      inline MuonBxCollection* getMuons(const unsigned int copy) override { return muons_[copy].get(); };
      inline MuonShowerBxCollection* getMuonShowers(const unsigned int copy) override {
        return muonShowers_[copy].get();
      };
      inline EGammaBxCollection* getEGammas(const unsigned int copy) override { return egammas_[copy].get(); };
      inline EtSumBxCollection* getEtSums(const unsigned int copy) override { return etsums_[copy].get(); };
      inline EtSumBxCollection* getZDCSums(const unsigned int copy) override { return zdcsums_[copy].get(); };
      inline JetBxCollection* getJets(const unsigned int copy) override { return jets_[copy].get(); };
      inline TauBxCollection* getTaus(const unsigned int copy) override { return taus_[copy].get(); };

      inline GlobalAlgBlkBxCollection* getAlgs() { return algBlk_.get(); };
      inline GlobalExtBlkBxCollection* getExts() { return extBlk_.get(); };

    private:
      std::array<std::unique_ptr<MuonBxCollection>, 6> muons_;
      std::array<std::unique_ptr<MuonShowerBxCollection>, 6> muonShowers_;
      std::array<std::unique_ptr<EGammaBxCollection>, 6> egammas_;
      std::array<std::unique_ptr<EtSumBxCollection>, 6> etsums_;
      std::array<std::unique_ptr<EtSumBxCollection>, 6> zdcsums_;
      std::array<std::unique_ptr<JetBxCollection>, 6> jets_;
      std::array<std::unique_ptr<TauBxCollection>, 6> taus_;

      std::unique_ptr<GlobalAlgBlkBxCollection> algBlk_;
      std::unique_ptr<GlobalExtBlkBxCollection> extBlk_;
    };
  }  // namespace stage2
}  // namespace l1t

#endif

#ifndef GMTTokens_h
#define GMTTokens_h

#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1Trigger/interface/Muon.h"

#include "CommonTokens.h"

namespace l1t {
   namespace stage2 {
      class GMTTokens : public CommonTokens {
         public:
            GMTTokens(const edm::ParameterSet&, edm::ConsumesCollector&);

            inline const edm::EDGetTokenT<RegionalMuonCandBxCollection>& getRegionalMuonCandTokenBMTF() const { return regionalMuonCandTokenBMTF_; };
            inline const edm::EDGetTokenT<RegionalMuonCandBxCollection>& getRegionalMuonCandTokenOMTF() const { return regionalMuonCandTokenOMTF_; };
            inline const edm::EDGetTokenT<RegionalMuonCandBxCollection>& getRegionalMuonCandTokenEMTF() const { return regionalMuonCandTokenEMTF_; };
            inline const edm::EDGetTokenT<MuonBxCollection>& getImdMuonTokenBMTF() const { return imdMuonTokenBMTF_; };
            inline const edm::EDGetTokenT<MuonBxCollection>& getImdMuonTokenEMTFNeg() const { return imdMuonTokenEMTFNeg_; };
            inline const edm::EDGetTokenT<MuonBxCollection>& getImdMuonTokenEMTFPos() const { return imdMuonTokenEMTFPos_; };
            inline const edm::EDGetTokenT<MuonBxCollection>& getImdMuonTokenOMTFNeg() const { return imdMuonTokenOMTFNeg_; };
            inline const edm::EDGetTokenT<MuonBxCollection>& getImdMuonTokenOMTFPos() const { return imdMuonTokenOMTFPos_; };

         private:
            edm::EDGetTokenT<RegionalMuonCandBxCollection> regionalMuonCandTokenBMTF_;
            edm::EDGetTokenT<RegionalMuonCandBxCollection> regionalMuonCandTokenOMTF_;
            edm::EDGetTokenT<RegionalMuonCandBxCollection> regionalMuonCandTokenEMTF_;
            edm::EDGetTokenT<MuonBxCollection> imdMuonTokenBMTF_;
            edm::EDGetTokenT<MuonBxCollection> imdMuonTokenEMTFNeg_;
            edm::EDGetTokenT<MuonBxCollection> imdMuonTokenEMTFPos_;
            edm::EDGetTokenT<MuonBxCollection> imdMuonTokenOMTFNeg_;
            edm::EDGetTokenT<MuonBxCollection> imdMuonTokenOMTFPos_;
      };
   }
}

#endif

#ifndef GMTTokens_h
#define GMTTokens_h

#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1Trigger/interface/Muon.h"

#include "EventFilter/L1TRawToDigi/interface/PackerTokens.h"

namespace l1t {
   namespace stage2 {
      class GMTTokens : public PackerTokens {
         public:
            GMTTokens(const edm::ParameterSet&, edm::ConsumesCollector&);

            inline const edm::EDGetTokenT<RegionalMuonCandBxCollection>& getRegionalMuonCandTokenBMTF() const { return regionalMuonCandTokenBMTF_; };
            inline const edm::EDGetTokenT<RegionalMuonCandBxCollection>& getRegionalMuonCandTokenOMTF() const { return regionalMuonCandTokenOMTF_; };
            inline const edm::EDGetTokenT<RegionalMuonCandBxCollection>& getRegionalMuonCandTokenEMTF() const { return regionalMuonCandTokenEMTF_; };
            inline const edm::EDGetTokenT<MuonBxCollection>& getMuonToken() const { return muonToken_; };

         private:
            edm::EDGetTokenT<RegionalMuonCandBxCollection> regionalMuonCandTokenBMTF_;
            edm::EDGetTokenT<RegionalMuonCandBxCollection> regionalMuonCandTokenOMTF_;
            edm::EDGetTokenT<RegionalMuonCandBxCollection> regionalMuonCandTokenEMTF_;
            edm::EDGetTokenT<MuonBxCollection> muonToken_;
      };
   }
}

#endif

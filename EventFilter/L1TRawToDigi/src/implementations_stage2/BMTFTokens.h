#ifndef BMTFTokens_h
#define BMTFTokens_h

#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
#include "EventFilter/L1TRawToDigi/interface/PackerTokens.h"

namespace l1t {
   namespace stage2 {
      class BMTFTokens : public PackerTokens {
         public:
           BMTFTokens(const edm::ParameterSet&, edm::ConsumesCollector&);

						inline const edm::EDGetTokenT<RegionalMuonCandBxCollection>& getOutputMuonToken() const {return outputMuonToken_;};
						inline const edm::EDGetTokenT<L1MuDTChambPhContainer>& getInputMuonTokenPh() const {return inputMuonTokenPh_;};
						inline const edm::EDGetTokenT<L1MuDTChambThContainer>& getInputMuonTokenTh() const {return inputMuonTokenTh_;};

         private:
						edm::EDGetTokenT<RegionalMuonCandBxCollection> outputMuonToken_;
						edm::EDGetTokenT<L1MuDTChambPhContainer> inputMuonTokenPh_;
						edm::EDGetTokenT<L1MuDTChambThContainer> inputMuonTokenTh_;

      };
   }
}

#endif

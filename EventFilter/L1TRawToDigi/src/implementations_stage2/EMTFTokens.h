
#ifndef EMTFTokens_h
#define EMTFTokens_h

/* #include "DataFormats/L1TMuon/interface/EMTFMuonCand.h" */
#include "DataFormats/L1TMuon/interface/EMTFOutput.h"

#include "EventFilter/L1TRawToDigi/interface/PackerTokens.h"

namespace l1t {
  namespace stage2 {
    class EMTFTokens : public PackerTokens {
    public:
      EMTFTokens(const edm::ParameterSet&, edm::ConsumesCollector&);
      
      /* inline const edm::EDGetTokenT<EMTFMuonCandBxCollection>& getEMTFMuonCandToken() const { return EMTFMuonCandToken_; }; */
      inline const edm::EDGetTokenT<EMTFOutputCollection>& getEMTFOutputToken() const { return EMTFOutputToken_; };
      
    private:
      
      /* edm::EDGetTokenT<EMTFMuonCandBxCollection> EMTFMuonCandToken_; */
      edm::EDGetTokenT<EMTFOutputCollection> EMTFOutputToken_;
      
    };
  }
}

#endif

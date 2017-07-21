
#ifndef EventFilter_L1TRawToDigi_EMTFTokens_h
#define EventFilter_L1TRawToDigi_EMTFTokens_h

#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1TMuon/interface/EMTFDaqOut.h"
#include "DataFormats/L1TMuon/interface/EMTFHit.h"
#include "DataFormats/L1TMuon/interface/EMTFTrack.h"
#include "DataFormats/L1TMuon/interface/EMTFHit2016.h"
#include "DataFormats/L1TMuon/interface/EMTFTrack2016.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"

#include "EventFilter/L1TRawToDigi/interface/PackerTokens.h"

namespace l1t {
  namespace stage2 {
    class EMTFTokens : public PackerTokens {
    public:
      EMTFTokens(const edm::ParameterSet&, edm::ConsumesCollector&);
      
      inline const edm::EDGetTokenT<RegionalMuonCandBxCollection>& getRegionalMuonCandToken() const { return regionalMuonCandToken_; }
      inline const edm::EDGetTokenT<EMTFDaqOutCollection>& getEMTFDaqOutToken() const { return EMTFDaqOutToken_; }
      inline const edm::EDGetTokenT<EMTFHitCollection>& getEMTFHitToken() const { return EMTFHitToken_; }
      inline const edm::EDGetTokenT<EMTFTrackCollection>& getEMTFTrackToken() const { return EMTFTrackToken_; }
      inline const edm::EDGetTokenT<EMTFHit2016Collection>& getEMTFHit2016Token() const { return EMTFHit2016Token_; }
      inline const edm::EDGetTokenT<EMTFTrack2016Collection>& getEMTFTrack2016Token() const { return EMTFTrack2016Token_; }
      inline const edm::EDGetTokenT<CSCCorrelatedLCTDigiCollection>& getEMTFLCTToken() const { return EMTFLCTToken_; }
      
    private:
      
      edm::EDGetTokenT<RegionalMuonCandBxCollection> regionalMuonCandToken_;
      edm::EDGetTokenT<EMTFDaqOutCollection> EMTFDaqOutToken_;
      edm::EDGetTokenT<EMTFHitCollection> EMTFHitToken_;
      edm::EDGetTokenT<EMTFTrackCollection> EMTFTrackToken_;
      edm::EDGetTokenT<EMTFHit2016Collection> EMTFHit2016Token_;
      edm::EDGetTokenT<EMTFTrack2016Collection> EMTFTrack2016Token_;
      edm::EDGetTokenT<CSCCorrelatedLCTDigiCollection> EMTFLCTToken_;
      
    };
  }
}

#endif /* define EventFilter_L1TRawToDigi_EMTFTokens_h */

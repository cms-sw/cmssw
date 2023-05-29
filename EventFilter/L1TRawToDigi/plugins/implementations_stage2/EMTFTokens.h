#ifndef EventFilter_L1TRawToDigi_EMTFTokens_h
#define EventFilter_L1TRawToDigi_EMTFTokens_h

#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonShower.h"
#include "DataFormats/L1TMuon/interface/EMTFDaqOut.h"
#include "DataFormats/L1TMuon/interface/EMTFHit.h"
#include "DataFormats/L1TMuon/interface/EMTFTrack.h"
#include "DataFormats/L1TMuon/interface/CPPFDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCShowerDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiClusterCollection.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "EventFilter/L1TRawToDigi/interface/PackerTokens.h"

namespace l1t {
  namespace stage2 {
    class EMTFTokens : public PackerTokens {
    public:
      EMTFTokens(const edm::ParameterSet&, edm::ConsumesCollector&);

      inline const edm::EDGetTokenT<RegionalMuonCandBxCollection>& getRegionalMuonCandToken() const {
        return regionalMuonCandToken_;
      }
      inline const edm::EDGetTokenT<RegionalMuonShowerBxCollection>& getRegionalMuonShowerToken() const {
        return regionalMuonShowerToken_;
      }
      inline const edm::EDGetTokenT<EMTFDaqOutCollection>& getEMTFDaqOutToken() const { return EMTFDaqOutToken_; }
      inline const edm::EDGetTokenT<EMTFHitCollection>& getEMTFHitToken() const { return EMTFHitToken_; }
      inline const edm::EDGetTokenT<EMTFTrackCollection>& getEMTFTrackToken() const { return EMTFTrackToken_; }
      inline const edm::EDGetTokenT<CSCCorrelatedLCTDigiCollection>& getEMTFLCTToken() const { return EMTFLCTToken_; }
      inline const edm::EDGetTokenT<CSCShowerDigiCollection>& getEMTFCSCShowerToken() const {
        return EMTFCSCShowerToken_;
      }
      inline const edm::EDGetTokenT<CPPFDigiCollection>& getEMTFCPPFToken() const { return EMTFCPPFToken_; }
      inline const edm::EDGetTokenT<GEMPadDigiClusterCollection>& getEMTFGEMPadClusterToken() const {
        return EMTFGEMPadClusterToken_;
      }

    private:
      edm::EDGetTokenT<RegionalMuonCandBxCollection> regionalMuonCandToken_;
      edm::EDGetTokenT<RegionalMuonShowerBxCollection> regionalMuonShowerToken_;
      edm::EDGetTokenT<EMTFDaqOutCollection> EMTFDaqOutToken_;
      edm::EDGetTokenT<EMTFHitCollection> EMTFHitToken_;
      edm::EDGetTokenT<EMTFTrackCollection> EMTFTrackToken_;
      edm::EDGetTokenT<CSCCorrelatedLCTDigiCollection> EMTFLCTToken_;
      edm::EDGetTokenT<CSCShowerDigiCollection> EMTFCSCShowerToken_;
      edm::EDGetTokenT<CPPFDigiCollection> EMTFCPPFToken_;
      edm::EDGetTokenT<GEMPadDigiClusterCollection> EMTFGEMPadClusterToken_;
    };
  }  // namespace stage2
}  // namespace l1t

#endif /* define EventFilter_L1TRawToDigi_EMTFTokens_h */

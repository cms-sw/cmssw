
#ifndef EventFilter_L1TRawToDigi_EMTFCollections_h
#define EventFilter_L1TRawToDigi_EMTFCollections_h

#include <iostream>  // For use in all EMTFBlock files
#include <iomanip>   // For things like std::setw

#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonShower.h"
#include "DataFormats/L1TMuon/interface/EMTFDaqOut.h"
#include "DataFormats/L1TMuon/interface/EMTFHit.h"
#include "DataFormats/L1TMuon/interface/EMTFTrack.h"
#include "DataFormats/L1TMuon/interface/CPPFDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCShowerDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiClusterCollection.h"

#include "EventFilter/L1TRawToDigi/interface/UnpackerCollections.h"

#include "L1Trigger/L1TMuonEndCap/interface/MicroGMTConverter.h"

namespace l1t {
  namespace stage2 {
    namespace L1TMuonEndCap =
        ::emtf;  // use alias 'L1TMuonEndCap' for the namespace 'emtf' used in L1Trigger/L1TMuonEndCap

    class EMTFCollections : public UnpackerCollections {
    public:
      EMTFCollections(edm::Event& e)
          : UnpackerCollections(e),  // What are these? - AWB 27.01.16
            regionalMuonCands_(new RegionalMuonCandBxCollection()),
            regionalMuonShowers_(new RegionalMuonShowerBxCollection()),
            EMTFDaqOuts_(new EMTFDaqOutCollection()),
            EMTFHits_(new EMTFHitCollection()),
            EMTFHits_ZS_(new EMTFHitCollection()),
            EMTFTracks_(new EMTFTrackCollection()),
            EMTFLCTs_(new CSCCorrelatedLCTDigiCollection()),
            EMTFCSCShowers_(new CSCShowerDigiCollection()),
            EMTFCPPFs_(new CPPFDigiCollection()),
            EMTFCPPFs_ZS_(new CPPFDigiCollection()),
            EMTFGEMPadClusters_(std::make_unique<GEMPadDigiClusterCollection>()),
            EMTFGEMPadClusters_ZS_(std::make_unique<GEMPadDigiClusterCollection>()){};

      ~EMTFCollections() override;

      inline RegionalMuonCandBxCollection* getRegionalMuonCands() { return regionalMuonCands_.get(); }
      inline RegionalMuonShowerBxCollection* getRegionalMuonShowers() { return regionalMuonShowers_.get(); }
      inline EMTFDaqOutCollection* getEMTFDaqOuts() { return EMTFDaqOuts_.get(); }
      inline EMTFHitCollection* getEMTFHits() { return EMTFHits_.get(); }
      inline EMTFHitCollection* getEMTFHits_ZS() { return EMTFHits_ZS_.get(); }
      inline EMTFTrackCollection* getEMTFTracks() { return EMTFTracks_.get(); }
      inline CSCCorrelatedLCTDigiCollection* getEMTFLCTs() { return EMTFLCTs_.get(); }
      inline CSCShowerDigiCollection* getEMTFCSCShowers() { return EMTFCSCShowers_.get(); }
      inline CPPFDigiCollection* getEMTFCPPFs() { return EMTFCPPFs_.get(); }
      inline CPPFDigiCollection* getEMTFCPPFs_ZS() { return EMTFCPPFs_ZS_.get(); }
      inline GEMPadDigiClusterCollection* getEMTFGEMPadClusters() { return EMTFGEMPadClusters_.get(); }
      inline GEMPadDigiClusterCollection* getEMTFGEMPadClusters_ZS() { return EMTFGEMPadClusters_ZS_.get(); }

    private:
      std::unique_ptr<RegionalMuonCandBxCollection> regionalMuonCands_;
      std::unique_ptr<RegionalMuonShowerBxCollection> regionalMuonShowers_;
      std::unique_ptr<EMTFDaqOutCollection> EMTFDaqOuts_;
      std::unique_ptr<EMTFHitCollection> EMTFHits_;
      std::unique_ptr<EMTFHitCollection> EMTFHits_ZS_;
      std::unique_ptr<EMTFTrackCollection> EMTFTracks_;
      std::unique_ptr<CSCCorrelatedLCTDigiCollection> EMTFLCTs_;
      std::unique_ptr<CSCShowerDigiCollection> EMTFCSCShowers_;
      std::unique_ptr<CPPFDigiCollection> EMTFCPPFs_;
      std::unique_ptr<CPPFDigiCollection> EMTFCPPFs_ZS_;
      std::unique_ptr<GEMPadDigiClusterCollection> EMTFGEMPadClusters_;
      std::unique_ptr<GEMPadDigiClusterCollection> EMTFGEMPadClusters_ZS_;
    };
  }  // namespace stage2
}  // namespace l1t

#endif

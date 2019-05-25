
#ifndef EventFilter_L1TRawToDigi_EMTFCollections_h
#define EventFilter_L1TRawToDigi_EMTFCollections_h

#include <iostream>  // For use in all EMTFBlock files
#include <iomanip>   // For things like std::setw

#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1TMuon/interface/EMTFDaqOut.h"
#include "DataFormats/L1TMuon/interface/EMTFHit.h"
#include "DataFormats/L1TMuon/interface/EMTFTrack.h"
#include "DataFormats/L1TMuon/interface/CPPFDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"

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
            EMTFDaqOuts_(new EMTFDaqOutCollection()),
            EMTFHits_(new EMTFHitCollection()),
            EMTFHits_ZS_(new EMTFHitCollection()),
            EMTFTracks_(new EMTFTrackCollection()),
            EMTFLCTs_(new CSCCorrelatedLCTDigiCollection()),
            EMTFCPPFs_(new CPPFDigiCollection()),
            EMTFCPPFs_ZS_(new CPPFDigiCollection()){};

      ~EMTFCollections() override;

      inline RegionalMuonCandBxCollection* getRegionalMuonCands() { return regionalMuonCands_.get(); }
      // How does this work?  I haven't even defined a "get()" function for the EMTFDaqOutCollection. - AWB 28.01.16
      inline EMTFDaqOutCollection* getEMTFDaqOuts() { return EMTFDaqOuts_.get(); }
      inline EMTFHitCollection* getEMTFHits() { return EMTFHits_.get(); }
      inline EMTFHitCollection* getEMTFHits_ZS() { return EMTFHits_ZS_.get(); }
      inline EMTFTrackCollection* getEMTFTracks() { return EMTFTracks_.get(); }
      inline CSCCorrelatedLCTDigiCollection* getEMTFLCTs() { return EMTFLCTs_.get(); }
      inline CPPFDigiCollection* getEMTFCPPFs() { return EMTFCPPFs_.get(); }
      inline CPPFDigiCollection* getEMTFCPPFs_ZS() { return EMTFCPPFs_ZS_.get(); }

    private:
      std::unique_ptr<RegionalMuonCandBxCollection> regionalMuonCands_;
      std::unique_ptr<EMTFDaqOutCollection> EMTFDaqOuts_;
      std::unique_ptr<EMTFHitCollection> EMTFHits_;
      std::unique_ptr<EMTFHitCollection> EMTFHits_ZS_;
      std::unique_ptr<EMTFTrackCollection> EMTFTracks_;
      std::unique_ptr<CSCCorrelatedLCTDigiCollection> EMTFLCTs_;
      std::unique_ptr<CPPFDigiCollection> EMTFCPPFs_;
      std::unique_ptr<CPPFDigiCollection> EMTFCPPFs_ZS_;
    };
  }  // namespace stage2
}  // namespace l1t

#endif


#ifndef EventFilter_L1TRawToDigi_EMTFCollections_h
#define EventFilter_L1TRawToDigi_EMTFCollections_h

#include <iostream> // For use in all EMTFBlock files
#include <iomanip>  // For things like std::setw

#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1TMuon/interface/EMTFDaqOut.h"
#include "DataFormats/L1TMuon/interface/EMTFHit.h"
#include "DataFormats/L1TMuon/interface/EMTFTrack.h"
#include "DataFormats/L1TMuon/interface/EMTFHit2016.h"
#include "DataFormats/L1TMuon/interface/EMTFTrack2016.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"

#include "EventFilter/L1TRawToDigi/interface/UnpackerCollections.h"

namespace l1t {
  namespace stage2 {
    class EMTFCollections : public UnpackerCollections {
    public:
    EMTFCollections(edm::Event& e) :
      UnpackerCollections(e), // What are these? - AWB 27.01.16
	regionalMuonCands_(new RegionalMuonCandBxCollection()),
	EMTFDaqOuts_(new EMTFDaqOutCollection()),
	EMTFHits_(new EMTFHitCollection()),
	EMTFTracks_(new EMTFTrackCollection()),
	EMTFHit2016s_(new EMTFHit2016Collection()),
	EMTFTrack2016s_(new EMTFTrack2016Collection()),
	EMTFLCTs_(new CSCCorrelatedLCTDigiCollection())
	  {};
      
      virtual ~EMTFCollections();
      
      inline RegionalMuonCandBxCollection* getRegionalMuonCands() { return regionalMuonCands_.get(); }
      // How does this work?  I haven't even defined a "get()" function for the EMTFDaqOutCollection. - AWB 28.01.16
      inline EMTFDaqOutCollection* getEMTFDaqOuts() { return EMTFDaqOuts_.get(); }
      inline EMTFHitCollection*    getEMTFHits()    { return EMTFHits_.get();    }       
      inline EMTFTrackCollection*  getEMTFTracks()  { return EMTFTracks_.get();  }       
      inline EMTFHit2016Collection*    getEMTFHit2016s()    { return EMTFHit2016s_.get();    }       
      inline EMTFTrack2016Collection*  getEMTFTrack2016s()  { return EMTFTrack2016s_.get();  }       
      inline CSCCorrelatedLCTDigiCollection* getEMTFLCTs() { return EMTFLCTs_.get(); }
      
    private:
      
      std::unique_ptr<RegionalMuonCandBxCollection> regionalMuonCands_;
      std::unique_ptr<EMTFDaqOutCollection>         EMTFDaqOuts_;
      std::unique_ptr<EMTFHitCollection>            EMTFHits_;
      std::unique_ptr<EMTFTrackCollection>          EMTFTracks_;
      std::unique_ptr<EMTFHit2016Collection>            EMTFHit2016s_;
      std::unique_ptr<EMTFTrack2016Collection>          EMTFTrack2016s_;
      std::unique_ptr<CSCCorrelatedLCTDigiCollection> EMTFLCTs_;
      
    };
  }
}

#endif

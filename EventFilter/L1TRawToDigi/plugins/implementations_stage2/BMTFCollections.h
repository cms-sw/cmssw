#ifndef BMTFCollections_h
#define BMTFCollections_h


#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "EventFilter/L1TRawToDigi/interface/UnpackerCollections.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
//#include "L1TObjectCollections.h"

namespace l1t {
   namespace stage2 {
     class BMTFCollections : public UnpackerCollections {
     public:
            BMTFCollections(edm::Event& e) :
               UnpackerCollections(e),
               outputMuons_ (new RegionalMuonCandBxCollection()),
	       outputMuons2_ (new RegionalMuonCandBxCollection()),
               inputMuonsPh_ (new L1MuDTChambPhContainer),
               inputMuonsTh_ (new L1MuDTChambThContainer)
            {};

            ~BMTFCollections() override;

	    inline RegionalMuonCandBxCollection* getBMTFMuons() {return outputMuons_.get();};
	    inline RegionalMuonCandBxCollection* getBMTF2Muons() {return outputMuons2_.get();};
	    inline L1MuDTChambPhContainer* getInMuonsPh() { return inputMuonsPh_.get(); };
	    inline L1MuDTChambThContainer* getInMuonsTh() { return inputMuonsTh_.get(); };

     private:
	    std::unique_ptr<RegionalMuonCandBxCollection> outputMuons_;
	    std::unique_ptr<RegionalMuonCandBxCollection> outputMuons2_;
	    std::unique_ptr<L1MuDTChambPhContainer> inputMuonsPh_;
	    std::unique_ptr<L1MuDTChambThContainer> inputMuonsTh_;
      };
   }
}

#endif

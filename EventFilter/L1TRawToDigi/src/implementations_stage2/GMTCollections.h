#ifndef GMTCollections_h
#define GMTCollections_h

#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1Trigger/interface/Muon.h"

//#include "EventFilter/L1TRawToDigi/interface/UnpackerCollections.h"
#include "L1TObjectCollections.h"

namespace l1t {
   namespace stage2 {
      class GMTCollections : public L1TObjectCollections {
         public:
            GMTCollections(edm::Event& e) :
               L1TObjectCollections(e),
               regionalMuonCandsBMTF_(new RegionalMuonCandBxCollection()),
               regionalMuonCandsOMTF_(new RegionalMuonCandBxCollection()),
               regionalMuonCandsEMTF_(new RegionalMuonCandBxCollection()),
               muons_(new MuonBxCollection()) {};

            virtual ~GMTCollections();

            inline RegionalMuonCandBxCollection* getRegionalMuonCandsBMTF() { return regionalMuonCandsBMTF_.get(); };
            inline RegionalMuonCandBxCollection* getRegionalMuonCandsOMTF() { return regionalMuonCandsOMTF_.get(); };
            inline RegionalMuonCandBxCollection* getRegionalMuonCandsEMTF() { return regionalMuonCandsEMTF_.get(); };
            inline MuonBxCollection* getMuons() { return muons_.get(); };

         private:
            std::auto_ptr<RegionalMuonCandBxCollection> regionalMuonCandsBMTF_;
            std::auto_ptr<RegionalMuonCandBxCollection> regionalMuonCandsOMTF_;
            std::auto_ptr<RegionalMuonCandBxCollection> regionalMuonCandsEMTF_;
            std::auto_ptr<MuonBxCollection> muons_;
      };
   }
}

#endif

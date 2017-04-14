#ifndef GMTCollections_h
#define GMTCollections_h

#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1Trigger/interface/Muon.h"

//#include "EventFilter/L1TRawToDigi/interface/UnpackerCollections.h"
#include "L1TObjectCollections.h"

#include <array>

namespace l1t {
   namespace stage2 {
      class GMTCollections : public L1TObjectCollections {
         public:
            GMTCollections(edm::Event& e) :
               L1TObjectCollections(e),
               regionalMuonCandsBMTF_(std::make_unique<RegionalMuonCandBxCollection>()),
               regionalMuonCandsOMTF_(std::make_unique<RegionalMuonCandBxCollection>()),
               regionalMuonCandsEMTF_(std::make_unique<RegionalMuonCandBxCollection>()),
               muons_(),
               imdMuonsBMTF_(std::make_unique<MuonBxCollection>()),
               imdMuonsEMTFNeg_(std::make_unique<MuonBxCollection>()),
               imdMuonsEMTFPos_(std::make_unique<MuonBxCollection>()),
               imdMuonsOMTFNeg_(std::make_unique<MuonBxCollection>()),
               imdMuonsOMTFPos_(std::make_unique<MuonBxCollection>())
            {
               std::generate(muons_.begin(), muons_.end(), []{ return std::make_unique<MuonBxCollection>(); });
            };

            virtual ~GMTCollections();

            inline RegionalMuonCandBxCollection* getRegionalMuonCandsBMTF() { return regionalMuonCandsBMTF_.get(); };
            inline RegionalMuonCandBxCollection* getRegionalMuonCandsOMTF() { return regionalMuonCandsOMTF_.get(); };
            inline RegionalMuonCandBxCollection* getRegionalMuonCandsEMTF() { return regionalMuonCandsEMTF_.get(); };
            inline MuonBxCollection* getMuons(const unsigned int copy) override { return muons_[copy].get(); };
            inline MuonBxCollection* getImdMuonsBMTF() { return imdMuonsBMTF_.get(); };
            inline MuonBxCollection* getImdMuonsEMTFNeg() { return imdMuonsEMTFNeg_.get(); };
            inline MuonBxCollection* getImdMuonsEMTFPos() { return imdMuonsEMTFPos_.get(); };
            inline MuonBxCollection* getImdMuonsOMTFNeg() { return imdMuonsOMTFNeg_.get(); };
            inline MuonBxCollection* getImdMuonsOMTFPos() { return imdMuonsOMTFPos_.get(); };

         private:
            std::unique_ptr<RegionalMuonCandBxCollection> regionalMuonCandsBMTF_;
            std::unique_ptr<RegionalMuonCandBxCollection> regionalMuonCandsOMTF_;
            std::unique_ptr<RegionalMuonCandBxCollection> regionalMuonCandsEMTF_;
            std::array<std::unique_ptr<MuonBxCollection>, 6> muons_;
            std::unique_ptr<MuonBxCollection> imdMuonsBMTF_;
            std::unique_ptr<MuonBxCollection> imdMuonsEMTFNeg_;
            std::unique_ptr<MuonBxCollection> imdMuonsEMTFPos_;
            std::unique_ptr<MuonBxCollection> imdMuonsOMTFNeg_;
            std::unique_ptr<MuonBxCollection> imdMuonsOMTFPos_;
      };
   }
}

#endif

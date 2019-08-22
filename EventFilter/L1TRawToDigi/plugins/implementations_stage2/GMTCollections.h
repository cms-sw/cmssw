#ifndef GMTCollections_h
#define GMTCollections_h

#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1Trigger/interface/Muon.h"

#include "L1TObjectCollections.h"

#include <array>

namespace l1t {
  namespace stage2 {
    class GMTCollections : public L1TObjectCollections {
    public:
      // If the zero suppression deletes all the blocks used to
      // fill a collection the BX range cannot be determined.
      // Set default values here to then create an empty collection
      // with a defined BX range.
      GMTCollections(
          edm::Event& e, const int iFirstBx = -2, const int iLastBx = 2, const int oFirstBx = -2, const int oLastBx = 2)
          : L1TObjectCollections(e),
            regionalMuonCandsBMTF_(std::make_unique<RegionalMuonCandBxCollection>(0, iFirstBx, iLastBx)),
            regionalMuonCandsOMTF_(std::make_unique<RegionalMuonCandBxCollection>(0, iFirstBx, iLastBx)),
            regionalMuonCandsEMTF_(std::make_unique<RegionalMuonCandBxCollection>(0, iFirstBx, iLastBx)),
            muons_(),
            imdMuonsBMTF_(std::make_unique<MuonBxCollection>(0, oFirstBx, oLastBx)),
            imdMuonsEMTFNeg_(std::make_unique<MuonBxCollection>(0, oFirstBx, oLastBx)),
            imdMuonsEMTFPos_(std::make_unique<MuonBxCollection>(0, oFirstBx, oLastBx)),
            imdMuonsOMTFNeg_(std::make_unique<MuonBxCollection>(0, oFirstBx, oLastBx)),
            imdMuonsOMTFPos_(std::make_unique<MuonBxCollection>(0, oFirstBx, oLastBx)) {
        std::generate(muons_.begin(), muons_.end(), [&oFirstBx, &oLastBx] {
          return std::make_unique<MuonBxCollection>(0, oFirstBx, oLastBx);
        });
      };

      ~GMTCollections() override;

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
  }  // namespace stage2
}  // namespace l1t

#endif

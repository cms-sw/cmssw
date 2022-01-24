#ifndef GMTCollections_h
#define GMTCollections_h

#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1Trigger/interface/Muon.h"

#include "DataFormats/L1TMuon/interface/RegionalMuonShower.h"
#include "DataFormats/L1Trigger/interface/MuonShower.h"

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
            imdMuonsOMTFPos_(std::make_unique<MuonBxCollection>(0, oFirstBx, oLastBx)),

            regionalMuonShowersEMTF_(std::make_unique<RegionalMuonShowerBxCollection>(0, iFirstBx, iLastBx)),
            muonShowers_() {
        std::generate(muons_.begin(), muons_.end(), [&oFirstBx, &oLastBx] {
          return std::make_unique<MuonBxCollection>(0, oFirstBx, oLastBx);
        });
        std::generate(muonShowers_.begin(), muonShowers_.end(), [&oFirstBx, &oLastBx] {
          return std::make_unique<MuonShowerBxCollection>(0, oFirstBx, oLastBx);
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

      inline RegionalMuonShowerBxCollection* getRegionalMuonShowersEMTF() { return regionalMuonShowersEMTF_.get(); };
      inline MuonShowerBxCollection* getMuonShowers(const unsigned int copy) override {
        return muonShowers_[copy].get();
      };

      static constexpr size_t NUM_OUTPUT_COPIES{6};

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

      std::unique_ptr<RegionalMuonShowerBxCollection> regionalMuonShowersEMTF_;
      std::array<std::unique_ptr<MuonShowerBxCollection>, 6> muonShowers_;
    };
  }  // namespace stage2
}  // namespace l1t

#endif

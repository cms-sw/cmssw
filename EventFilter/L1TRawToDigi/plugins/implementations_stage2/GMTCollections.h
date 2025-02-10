#ifndef GMTCollections_h
#define GMTCollections_h

#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1Trigger/interface/Muon.h"

#include "DataFormats/L1TMuon/interface/RegionalMuonShower.h"
#include "DataFormats/L1Trigger/interface/MuonShower.h"

#include "L1TObjectCollections.h"
#include "GMTPutTokens.h"

#include <array>

namespace l1t {
  namespace stage2 {
    class GMTCollections : public L1TObjectCollections {
    public:
      // If the zero suppression deletes all the blocks used to
      // fill a collection the BX range cannot be determined.
      // Set default values here to then create an empty collection
      // with a defined BX range.
      GMTCollections(edm::Event& e,
                     GMTPutTokens const& iTokens,
                     const int iFirstBx = -2,
                     const int iLastBx = 2,
                     const int oFirstBx = -2,
                     const int oLastBx = 2)
          : L1TObjectCollections(e),
            regionalMuonCandsBMTF_(0, iFirstBx, iLastBx),
            regionalMuonCandsOMTF_(0, iFirstBx, iLastBx),
            regionalMuonCandsEMTF_(0, iFirstBx, iLastBx),
            muons_{{{0, oFirstBx, oLastBx},
                    {0, oFirstBx, oLastBx},
                    {0, oFirstBx, oLastBx},
                    {0, oFirstBx, oLastBx},
                    {0, oFirstBx, oLastBx},
                    {0, oFirstBx, oLastBx}}},
            imdMuonsBMTF_(0, oFirstBx, oLastBx),
            imdMuonsEMTFNeg_(0, oFirstBx, oLastBx),
            imdMuonsEMTFPos_(0, oFirstBx, oLastBx),
            imdMuonsOMTFNeg_(0, oFirstBx, oLastBx),
            imdMuonsOMTFPos_(0, oFirstBx, oLastBx),
            regionalMuonShowersEMTF_(0, iFirstBx, iLastBx),
            muonShowers_{{{0, oFirstBx, oLastBx},
                          {0, oFirstBx, oLastBx},
                          {0, oFirstBx, oLastBx},
                          {0, oFirstBx, oLastBx},
                          {0, oFirstBx, oLastBx},
                          {0, oFirstBx, oLastBx}}},
            tokens_(iTokens) {};

      ~GMTCollections() override;

      inline RegionalMuonCandBxCollection* getRegionalMuonCandsBMTF() { return &regionalMuonCandsBMTF_; };
      inline RegionalMuonCandBxCollection* getRegionalMuonCandsOMTF() { return &regionalMuonCandsOMTF_; };
      inline RegionalMuonCandBxCollection* getRegionalMuonCandsEMTF() { return &regionalMuonCandsEMTF_; };
      inline MuonBxCollection* getMuons(const unsigned int copy) override { return &muons_[copy]; };
      inline MuonBxCollection* getImdMuonsBMTF() { return &imdMuonsBMTF_; };
      inline MuonBxCollection* getImdMuonsEMTFNeg() { return &imdMuonsEMTFNeg_; };
      inline MuonBxCollection* getImdMuonsEMTFPos() { return &imdMuonsEMTFPos_; };
      inline MuonBxCollection* getImdMuonsOMTFNeg() { return &imdMuonsOMTFNeg_; };
      inline MuonBxCollection* getImdMuonsOMTFPos() { return &imdMuonsOMTFPos_; };

      inline RegionalMuonShowerBxCollection* getRegionalMuonShowersEMTF() { return &regionalMuonShowersEMTF_; };
      inline MuonShowerBxCollection* getMuonShowers(const unsigned int copy) override { return &muonShowers_[copy]; };

      static constexpr size_t NUM_OUTPUT_COPIES{6};

    private:
      RegionalMuonCandBxCollection regionalMuonCandsBMTF_;
      RegionalMuonCandBxCollection regionalMuonCandsOMTF_;
      RegionalMuonCandBxCollection regionalMuonCandsEMTF_;
      std::array<MuonBxCollection, 6> muons_;
      MuonBxCollection imdMuonsBMTF_;
      MuonBxCollection imdMuonsEMTFNeg_;
      MuonBxCollection imdMuonsEMTFPos_;
      MuonBxCollection imdMuonsOMTFNeg_;
      MuonBxCollection imdMuonsOMTFPos_;

      RegionalMuonShowerBxCollection regionalMuonShowersEMTF_;
      std::array<MuonShowerBxCollection, 6> muonShowers_;
      GMTPutTokens tokens_;
    };
  }  // namespace stage2
}  // namespace l1t

#endif

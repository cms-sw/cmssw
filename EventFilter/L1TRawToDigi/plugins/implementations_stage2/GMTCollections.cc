#include "FWCore/Framework/interface/Event.h"

#include "GMTCollections.h"

#include <string>

namespace l1t {
  namespace stage2 {
    GMTCollections::~GMTCollections() {
      event_.emplace(tokens_.bmtf_, std::move(regionalMuonCandsBMTF_));
      event_.emplace(tokens_.omtf_, std::move(regionalMuonCandsOMTF_));
      event_.emplace(tokens_.emtf_, std::move(regionalMuonCandsEMTF_));
      event_.emplace(tokens_.muon_, std::move(muons_[0]));
      assert(NUM_OUTPUT_COPIES == tokens_.muonCopies_.size());
      for (size_t i = 1; i < NUM_OUTPUT_COPIES; ++i) {
        event_.emplace(tokens_.muonCopies_[i], std::move(muons_[i]));
      }
      event_.emplace(tokens_.imdMuonsBMTF_, std::move(imdMuonsBMTF_));
      event_.emplace(tokens_.imdMuonsEMTFNeg_, std::move(imdMuonsEMTFNeg_));
      event_.emplace(tokens_.imdMuonsEMTFPos_, std::move(imdMuonsEMTFPos_));
      event_.emplace(tokens_.imdMuonsOMTFNeg_, std::move(imdMuonsOMTFNeg_));
      event_.emplace(tokens_.imdMuonsOMTFPos_, std::move(imdMuonsOMTFPos_));

      event_.emplace(tokens_.showerEMTF_, std::move(regionalMuonShowersEMTF_));
      event_.emplace(tokens_.muonShower_, std::move(muonShowers_[0]));
      assert(tokens_.muonShowerCopy_.size() == NUM_OUTPUT_COPIES);
      for (size_t i = 1; i < NUM_OUTPUT_COPIES; ++i) {
        event_.emplace(tokens_.muonShowerCopy_[i], std::move(muonShowers_[i]));
      }
    }
  }  // namespace stage2
}  // namespace l1t

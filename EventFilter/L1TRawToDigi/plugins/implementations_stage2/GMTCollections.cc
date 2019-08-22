#include "FWCore/Framework/interface/Event.h"

#include "GMTCollections.h"

#include <string>

namespace l1t {
  namespace stage2 {
    GMTCollections::~GMTCollections() {
      event_.put(std::move(regionalMuonCandsBMTF_), "BMTF");
      event_.put(std::move(regionalMuonCandsOMTF_), "OMTF");
      event_.put(std::move(regionalMuonCandsEMTF_), "EMTF");
      event_.put(std::move(muons_[0]), "Muon");
      for (int i = 1; i < 6; ++i) {
        event_.put(std::move(muons_[i]), "MuonCopy" + std::to_string(i));
      }
      event_.put(std::move(imdMuonsBMTF_), "imdMuonsBMTF");
      event_.put(std::move(imdMuonsEMTFNeg_), "imdMuonsEMTFNeg");
      event_.put(std::move(imdMuonsEMTFPos_), "imdMuonsEMTFPos");
      event_.put(std::move(imdMuonsOMTFNeg_), "imdMuonsOMTFNeg");
      event_.put(std::move(imdMuonsOMTFPos_), "imdMuonsOMTFPos");
    }
  }  // namespace stage2
}  // namespace l1t

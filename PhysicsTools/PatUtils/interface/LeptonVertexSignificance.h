//
//

#ifndef PhysicsTools_PatUtils_LeptonVertexSignificance_h
#define PhysicsTools_PatUtils_LeptonVertexSignificance_h

/**
  \class    pat::LeptonVertexSignificance LeptonVertexSignificance.h "PhysicsTools/PatUtils/interface/LeptonVertexSignificance.h"
  \brief    Calculates a lepton's vertex association significance

   LeptonVertexSignificance calculates the significance of the association
   of the lepton to a given vertex, as defined in CMS Note 2006/024

  \author   Steven Lowette
*/

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "FWCore/Utilities/interface/InputTag.h"

class TransientTrackBuilder;

namespace reco {
  class Track;
}

namespace pat {
  class Electron;
  class Muon;

  class LeptonVertexSignificance {
  public:
    LeptonVertexSignificance() = default;
    ~LeptonVertexSignificance() = default;

    //NOTE: expects vertices from "offlinePrimaryVerticesFromCTFTracks"
    static edm::InputTag vertexCollectionTag();

    //NOTE: expects TransientTrackBuilder to be a copy of one from record TransientTrackRecord with label "TransientTrackBuilder"
    float calculate(const Electron& anElectron,
                    const reco::VertexCollection& vertices,
                    const TransientTrackBuilder& builder);
    float calculate(const Muon& aMuon, const reco::VertexCollection& vertices, const TransientTrackBuilder& builder);

  private:
    float calculate(const reco::Track& track,
                    const reco::VertexCollection& vertices,
                    const TransientTrackBuilder& builder);
  };

}  // namespace pat

#endif

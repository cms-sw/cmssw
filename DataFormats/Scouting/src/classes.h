#include "DataFormats/Scouting/interface/ScoutingCaloJet.h"
#include "DataFormats/Scouting/interface/ScoutingPFJet.h"
#include "DataFormats/Scouting/interface/ScoutingParticle.h"
#include "DataFormats/Scouting/interface/ScoutingVertex.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/Ref.h"


namespace DataFormats_Scouting {
  struct dictionary {
    edm::Wrapper<ScoutingCaloJetCollection> sc1;
    edm::Wrapper<ScoutingParticleCollection> sc2;
    edm::Wrapper<ScoutingPFJetCollection> sc3;
    edm::Wrapper<ScoutingVertexCollection> sc4;
  };
}

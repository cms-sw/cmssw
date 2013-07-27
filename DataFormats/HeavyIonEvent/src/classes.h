#include "DataFormats/HeavyIonEvent/interface/Centrality.h"
#include "DataFormats/HeavyIonEvent/interface/EvtPlane.h"
#include "DataFormats/HeavyIonEvent/interface/HeavyIon.h"
#include "DataFormats/HeavyIonEvent/interface/VoronoiBackground.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace { 
  struct dictionary {

    reco::EvtPlane dummy1;
    edm::Wrapper<reco::EvtPlane> dummy2;

    reco::Centrality dummy3;
    edm::Wrapper<reco::Centrality> dummy4;

     reco::CentralityCollection ccol;
     edm::Wrapper<reco::CentralityCollection> wccol;

     reco::EvtPlaneCollection evcol;
     edm::Wrapper<reco::EvtPlaneCollection> wevcol;

     reco::VoronoiBackground vor;
     edm::Wrapper<reco::VoronoiBackground> wvor;
     reco::VoronoiBackgroundMap vormap;
     edm::Wrapper<reco::VoronoiBackgroundMap> wvormap;

     edm::AssociationMap<edm::OneToValue<std::vector<reco::PFCandidate>, reco::VoronoiBackground, unsigned int > > y1;
     edm::Wrapper<edm::AssociationMap<edm::OneToValue<std::vector<reco::PFCandidate>, reco::VoronoiBackground, unsigned int > > > y2;

     edm::AssociationMap<edm::OneToValue<edm::View<reco::Candidate>, reco::VoronoiBackground, unsigned int > > y3;
     edm::Wrapper<edm::AssociationMap<edm::OneToValue<edm::View<reco::Candidate>, reco::VoronoiBackground, unsigned int > > > y4;

     reco::PFVoronoiBackgroundMap pfvormap;
     edm::Wrapper<reco::PFVoronoiBackgroundMap> pfwvormap;

     edm::Wrapper<pat::HeavyIon >              w_v_p_hi;

  };
}



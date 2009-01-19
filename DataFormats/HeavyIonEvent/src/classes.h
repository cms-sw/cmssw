#include "DataFormats/HeavyIonEvent/interface/Centrality.h"
#include "DataFormats/HeavyIonEvent/interface/EvtPlane.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace { 
  struct dictionary {

    reco::EvtPlane dummy1;
    edm::Wrapper<reco::EvtPlane> dummy2;

    reco::Centrality dummy3;
    edm::Wrapper<reco::Centrality> dummy4;

  };
}



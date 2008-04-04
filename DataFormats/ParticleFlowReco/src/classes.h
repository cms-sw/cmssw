#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/Ref.h"
#include <DataFormats/Common/interface/OwnVector.h>
#include <DataFormats/Common/interface/ClonePolicy.h>

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "Math/Cartesian3D.h" 
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "Math/Polar3D.h" 
#include "Math/CylindricalEta3D.h" 
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/GsfPFRecTrack.h"
#include "Math/GenVector/PositionVector3D.h" 
#include "DataFormats/Math/interface/Point3D.h" 
#include "Rtypes.h" 
#include "DataFormats/Math/interface/Vector3D.h" 
#include "Math/PxPyPzE4D.h" 
#include "DataFormats/DetId/interface/DetId.h" 
#include <boost/cstdint.hpp> 
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h" 
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackExtra.h" 
/* #include "TrackingTools/TransientTrack/interface/TrackTransientTrack.h" */
/* #include "TrackingTools/TransientTrack/interface/GsfTransientTrack.h" */
#include "DataFormats/ParticleFlowReco/interface/PFSimParticle.h"
#include "DataFormats/ParticleFlowReco/interface/PFTrajectoryPoint.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementGsfTrack.h"  //Daniele
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementBrem.h"  //Daniele
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFNuclearInteraction.h"
#include "DataFormats/ParticleFlowReco/interface/PFConversion.h"
#include "DataFormats/ParticleFlowReco/interface/PFConversionFwd.h"


namespace { 
  namespace {

    std::vector<reco::PFCluster>                         dummy1;
    edm::Wrapper< std::vector<reco::PFCluster> >         dummy2;

    std::vector<reco::PFRecHit>                          dummy3;
    edm::Ref< std::vector<reco::PFRecHit> >              dummy4;
    edm::Wrapper< std::vector<reco::PFRecHit> >          dummy5;

    std::vector<reco::PFRecTrack>                        dummy6;
    edm::Wrapper< std::vector<reco::PFRecTrack> >        dummy7;

    std::vector<reco::GsfPFRecTrack>                     dummy6a;
    edm::Wrapper< std::vector<reco::GsfPFRecTrack> >     dummy7a;

/*     edm::RefToBase<reco::Track>                          dummy7b; */
/*     edm::reftobase::BaseHolder<reco::Track>              dummy7c; */
/*     edm::reftobase::Holder<reco::GsfTrack>               dummy7d;  */
/*     edm::Holder<reco::TrackTransientTrack>               dummy7e;  */
/*     edm::Holder<reco::GsfTransientTrack>                 dummy7f; */

    std::vector<reco::PFTrajectoryPoint>                 dummy8;
    edm::Wrapper< std::vector<reco::PFTrajectoryPoint> > dummy9;

    std::vector<reco::PFSimParticle>                     dummy10;
    edm::Wrapper< std::vector<reco::PFSimParticle> >     dummy11;

    edm::OwnVector< reco::PFBlockElement, 
      edm::ClonePolicy<reco::PFBlockElement> >           dummy12;
    edm::Wrapper< edm::OwnVector< reco::PFBlockElement,
      edm::ClonePolicy< reco::PFBlockElement> >  >       dummy13;
    std::vector<reco::PFBlockElement*>                   dummy14;

    std::vector<reco::PFBlock>                           dummy15;
    edm::Wrapper< std::vector<reco::PFBlock> >           dummy16;
    edm::Ref< std::vector<reco::PFBlock>, reco::PFBlock, edm::refhelper::FindUsingAdvance< std::vector<reco::PFBlock>, reco::PFBlock> >  dummy18;
    edm::Ref<std::vector<reco::PFRecTrack>,reco::PFRecTrack,edm::refhelper::FindUsingAdvance<std::vector<reco::PFRecTrack>,reco::PFRecTrack> > dummy19;
    edm::RefVector<std::vector<reco::PFRecTrack>,reco::PFRecTrack,edm::refhelper::FindUsingAdvance<std::vector<reco::PFRecTrack>,reco::PFRecTrack> > dummy19b;
    edm::Ref<std::vector<reco::GsfPFRecTrack>,reco::GsfPFRecTrack,edm::refhelper::FindUsingAdvance<std::vector<reco::GsfPFRecTrack>,reco::GsfPFRecTrack> > dummy19c;
    edm::RefVector<std::vector<reco::GsfPFRecTrack>,reco::GsfPFRecTrack,edm::refhelper::FindUsingAdvance<std::vector<reco::GsfPFRecTrack>,reco::GsfPFRecTrack> > dummy19d;
    edm::Ref<std::vector<reco::PFCluster>,reco::PFCluster,edm::refhelper::FindUsingAdvance<std::vector<reco::PFCluster>,reco::PFCluster> > dummy20;

    /* NuclearInteraction stuffs  */
    reco::PFNuclearInteraction                                dummy21;
    std::vector<reco::PFNuclearInteraction>                   dummy22;
    edm::Wrapper<std::vector<reco::PFNuclearInteraction> >    dummy23;
    edm::Ref<std::vector<reco::PFNuclearInteraction> >        dummy24;
    edm::RefProd<std::vector<reco::PFNuclearInteraction> >    dummy25;
    edm::RefVector<std::vector<reco::PFNuclearInteraction> >  dummy26;

    reco::PFConversionCollection dummy27;
    edm::Wrapper<reco::PFConversionCollection> dummy28; 
    edm::Ref<reco::PFConversionCollection> dummy29;
    edm::RefProd<reco::PFConversionCollection> dummy30;
    edm::Wrapper<edm::RefVector<reco::PFConversionCollection> > dummy31; 
    std::vector<edm::Ref<std::vector<reco::PFRecTrack>,reco::PFRecTrack,edm::refhelper::FindUsingAdvance<std::vector<reco::PFRecTrack>,reco::PFRecTrack> > > dummy32;


  }
}

#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/Ref.h"
#include <DataFormats/Common/interface/OwnVector.h>
#include <DataFormats/Common/interface/ClonePolicy.h>
#include "DataFormats/Common/interface/ValueMap.h"

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
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementSuperCluster.h" //Florian
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedTrackerVertex.h"
#include "DataFormats/ParticleFlowReco/interface/PFNuclearInteraction.h"
#include "DataFormats/ParticleFlowReco/interface/PFConversion.h"
#include "DataFormats/ParticleFlowReco/interface/PFConversionFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFV0.h"
#include "DataFormats/ParticleFlowReco/interface/ConvBremSeed.h"
#include "DataFormats/ParticleFlowReco/interface/ConvBremSeedFwd.h"
//Includes by Jamie
#include "DataFormats/ParticleFlowReco/interface/Calibratable.h"
#include "DataFormats/ParticleFlowReco/interface/CalibrationResultWrapper.h"
#include "DataFormats/ParticleFlowReco/interface/CalibrationProvenance.h"
#include "DataFormats/ParticleFlowReco/interface/CaloWindow.h"
#include "DataFormats/ParticleFlowReco/interface/CaloEllipse.h"
#include "DataFormats/ParticleFlowReco/interface/CaloBox.h"
#include "DataFormats/ParticleFlowReco/interface/ParticleFiltrationDecision.h"

#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertexCandidate.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertexSeed.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertex.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertexCandidateFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertexSeedFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertexFwd.h"

#include "DataFormats/ParticleFlowReco/interface/PreId.h"
#include "DataFormats/ParticleFlowReco/interface/PreIdFwd.h"
#include "DataFormats/ParticleFlowReco/interface/RecoPFClusterRefCandidate.h"
#include "DataFormats/ParticleFlowReco/interface/RecoPFClusterRefCandidateFwd.h"

#include <map>

namespace {
  struct dictionary {

    std::vector<reco::PFCluster>                         dummy1;
    edm::Wrapper< std::vector<reco::PFCluster> >         dummy2;
    
    reco::PFCluster::EEtoPSAssociation                   dummy1_x;
    reco::PFCluster::EEtoPSAssociation::value_type       dummy1_xx;
    edm::Wrapper< reco::PFCluster::EEtoPSAssociation >   dummy1_y;

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
    std::vector<edm::Ref<std::vector<reco::GsfPFRecTrack>,reco::GsfPFRecTrack,edm::refhelper::FindUsingAdvance<std::vector<reco::GsfPFRecTrack>,reco::GsfPFRecTrack> > > dummy19e;
    edm::Ref<std::vector<reco::PFCluster>,reco::PFCluster,edm::refhelper::FindUsingAdvance<std::vector<reco::PFCluster>,reco::PFCluster> > dummy20;



    reco::RecoPFClusterRefCandidateCollection vpfcr1;
    edm::Wrapper<reco::RecoPFClusterRefCandidateCollection> wpfcr1;
    edm::Ref<reco::RecoPFClusterRefCandidateCollection> rpfcrr1;
    edm::RefProd<reco::RecoPFClusterRefCandidateCollection> rpfcrpr1;
    edm::RefVector<reco::RecoPFClusterRefCandidateCollection> rvpfcrr1;


    edm::reftobase::Holder<reco::Candidate, reco::RecoPFClusterRefCandidateRef> rbpfr1;
    edm::reftobase::RefHolder<reco::RecoPFClusterRefCandidateRef> rbpfr2;
    edm::reftobase::VectorHolder<reco::Candidate, reco::RecoPFClusterRefCandidateRefVector> rbpfr3;
    edm::reftobase::RefVectorHolder<reco::RecoPFClusterRefCandidateRefVector> rbpfr4;


    /* Calibratable bits */
    pftools::Calibratable dummy50;
    std::vector<pftools::Calibratable> dummy51;
    pftools::CalibrationResultWrapper dummy52;
    pftools::CalibratableElement dummy53;
    pftools::CandidateWrapper dumm54;
    edm::Wrapper<std::vector<pftools::Calibratable> > dummy678;
    std::vector<pftools::CalibratableElement> dummy55;
    std::vector<pftools::CandidateWrapper> dummy56;
    std::vector<pftools::CalibrationResultWrapper> dummy57;
    pftools::CaloWindow dummy59;
    pftools::TestCaloWindow dummy59a;
    pftools::CaloRing dummy60;
    pftools::CaloEllipse dummy63;
    pftools::CaloBox dummy64;
    std::map<unsigned, pftools::CaloRing > dummy61;
    std::pair<unsigned, pftools::CaloRing > dummy61a;
    std::vector<pftools::CaloRing> dummy62;

    /*For testbeam analysis and "isolated" particle extraction */
    pftools::ParticleFiltrationDecision pfd;
    pftools::ParticleFiltrationDecisionCollection pfdColl;
    edm::Wrapper<pftools::ParticleFiltrationDecisionCollection> pfdCollWrapper;
    edm::Wrapper<pftools::ParticleFiltrationDecision> pfdWrapper;

    /* For nuclear interactions / displaced vertex finder */
    std::map<unsigned int,reco::PFDisplacedVertexCandidate::VertexLink> dummy1000;
    reco::PFDisplacedVertexCandidate::VertexLink dummy1001;
    reco::PFDisplacedVertexCandidate::DistMap dummy1002;
    reco::PFDisplacedVertexCandidate dummy1003;

    std::vector<reco::PFDisplacedVertexCandidate> dummy1004;
    edm::Wrapper<std::vector<reco::PFDisplacedVertexCandidate> > dummy1005;
    edm::Ref< std::vector<reco::PFDisplacedVertexCandidate>, reco::PFDisplacedVertexCandidate, edm::refhelper::FindUsingAdvance<std::vector<reco::PFDisplacedVertexCandidate>,reco::PFDisplacedVertexCandidate> > dummy1006;


   
    reco::PFDisplacedVertex dummy1007;

    std::vector<reco::PFDisplacedVertex> dummy1008;
    edm::Wrapper<std::vector<reco::PFDisplacedVertex> > dummy1009;
    edm::Ref< std::vector<reco::PFDisplacedVertex>, reco::PFDisplacedVertex, edm::refhelper::FindUsingAdvance<std::vector<reco::PFDisplacedVertex>,reco::PFDisplacedVertex> > dummy1010;

    std::vector<reco::PFDisplacedVertex::VertexTrackType> dummy1011;

    std::vector<std::pair<std::pair<unsigned int,unsigned int>,std::pair<unsigned int,unsigned int> > > dummy1012;

    std::pair<std::pair<unsigned int,unsigned int>,std::pair<unsigned int,unsigned int> > dummy1013;


    // Glowinski & Gouzevitch
    reco::PFMultiLinksTC dummy1014;
    // !Glowinski & Gouzevitch

    /* For PreID */
    reco::PreId dummy81;
    std::vector<reco::PreId> dummy82;
    edm::Ref< std::vector<reco::PreId> >  dummy85;
    edm::Wrapper<std::vector<reco::PreId> > dummy83;
    edm::Wrapper<edm::ValueMap<reco::PreIdRef> > dummy84;
    edm::Ref<std::vector<reco::PreId>,reco::PreId,edm::refhelper::FindUsingAdvance<std::vector<reco::PreId>,reco::PreId> > dummy86;


    /* PFBlockElementSuperCluster */
    reco::PFBlockElementSuperCluster dummy333;
    std::vector<reco::PFBlockElementSuperCluster> dummy334;
    edm::Wrapper<std::vector<reco::PFBlockElementSuperCluster> > dummy335;

    edm::RefVector<std::vector<reco::PFBlock>, reco::PFBlock, edm::refhelper::FindUsingAdvance< std::vector<reco::PFBlock>, reco::PFBlock> > dummy336;

edm::Wrapper<edm::RefVector<std::vector<reco::RecoPFClusterRefCandidate>,reco::RecoPFClusterRefCandidate,edm::refhelper::FindUsingAdvance<std::vector<reco::RecoPFClusterRefCandidate>,reco::RecoPFClusterRefCandidate> > > tpaaapfc2;


  };
}

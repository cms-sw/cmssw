#include <iostream>
#include <vector>
#include <memory>

// Framework
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
//
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
//
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TrackTransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionTrackEcalImpactPoint.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionTrackPairFinder.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionVertexFinder.h"

#include "DataFormats/EgammaTrackReco/interface/TrackCaloClusterAssociation.h"
#include "RecoEgamma/EgammaPhotonProducers/interface/SoftConversionProducer.h"

#include "Math/GenVector/VectorUtil.h"

bool trackQualityCut(reco::TransientTrack& trk){
  return (trk.numberOfValidHits() >= 3 && trk.normalizedChi2() >= 0.0);
}


SoftConversionProducer::SoftConversionProducer(const edm::ParameterSet& config) : conf_(config) {

  LogDebug("SoftConversionProducer") << " SoftConversionProducer CTOR " << "\n";
  
  conversionOITrackProducer_              = conf_.getParameter<std::string>("conversionOITrackProducer");
  conversionIOTrackProducer_              = conf_.getParameter<std::string>("conversionIOTrackProducer");
  outInTrackClusterAssociationCollection_ = conf_.getParameter<std::string>("outInTrackClusterAssociationCollection");
  inOutTrackClusterAssociationCollection_ = conf_.getParameter<std::string>("inOutTrackClusterAssociationCollection");
  clusterType_                            = conf_.getParameter<std::string>("clusterType");
  clusterProducer_                        = conf_.getParameter<std::string>("clusterProducer");
  clusterBarrelCollection_                = conf_.getParameter<std::string>("clusterBarrelCollection");
  clusterEndcapCollection_                = conf_.getParameter<std::string>("clusterEndcapCollection");
  softConversionCollection_               = conf_.getParameter<std::string>("softConversionCollection");
  
  theTrackPairFinder_ = 0;
  theVertexFinder_ = 0;
  theEcalImpactPositionFinder_ = 0;
  
  // Register the product
  produces< reco::ConversionCollection >(softConversionCollection_);

}


SoftConversionProducer::~SoftConversionProducer() {

  if(theTrackPairFinder_) delete theTrackPairFinder_;
  if(theVertexFinder_) delete theVertexFinder_;
  if(theEcalImpactPositionFinder_) delete theEcalImpactPositionFinder_; 

}


void  SoftConversionProducer::beginJob (edm::EventSetup const & theEventSetup) {
  
  //get magnetic field
  theEventSetup.get<IdealMagneticFieldRecord>().get(theMF_);  
  
  // instantiate the Track Pair Finder algorithm
  theTrackPairFinder_ = new ConversionTrackPairFinder ();

  // instantiate the Vertex Finder algorithm
  theVertexFinder_ = new ConversionVertexFinder ();
  
  // instantiate the algorithm for finding the position of the track extrapolation at the Ecal front face
  theEcalImpactPositionFinder_ = new   ConversionTrackEcalImpactPoint ( &(*theMF_) );

}


void  SoftConversionProducer::endJob () {}


void SoftConversionProducer::produce(edm::Event& theEvent, const edm::EventSetup& theEventSetup) {
  
  edm::Handle<reco::TrackCollection> outInTrkHandle;
  theEvent.getByLabel(conversionOITrackProducer_,  outInTrkHandle);
   
  edm::Handle<reco::TrackCaloClusterPtrAssociation> outInTrkClusterAssocHandle;
  theEvent.getByLabel( conversionOITrackProducer_, outInTrackClusterAssociationCollection_, outInTrkClusterAssocHandle);

  edm::Handle<reco::TrackCollection> inOutTrkHandle;
  theEvent.getByLabel(conversionIOTrackProducer_, inOutTrkHandle);
  
  edm::Handle<reco::TrackCaloClusterPtrAssociation> inOutTrkClusterAssocHandle;
  theEvent.getByLabel( conversionIOTrackProducer_, inOutTrackClusterAssociationCollection_, inOutTrkClusterAssocHandle);

  edm::Handle<edm::View<reco::CaloCluster> > clusterBarrelHandle;
  theEvent.getByLabel(clusterProducer_, clusterBarrelCollection_, clusterBarrelHandle);
    
  edm::Handle<edm::View<reco::CaloCluster> > clusterEndcapHandle;
  if(clusterType_ == "BasicCluster"){
    theEvent.getByLabel(clusterProducer_, clusterEndcapCollection_, clusterEndcapHandle);
  }

  // create temporary map to loop over tracks conveniently
  TrackClusterMap trackClusterMap;

  int nTracksOI = (int) outInTrkHandle->size();
  for(int itrk=0; itrk<nTracksOI; itrk++){
    reco::TrackRef tRef(outInTrkHandle,itrk);
    reco::CaloClusterPtr cRef = (*outInTrkClusterAssocHandle)[tRef];
    trackClusterMap.push_back(make_pair(tRef,cRef));
  }

  int nTracksIO = (int) inOutTrkHandle->size();
  for(int itrk=0; itrk<nTracksIO; itrk++){
    reco::TrackRef tRef(inOutTrkHandle,itrk);
    reco::CaloClusterPtr cRef = (*inOutTrkClusterAssocHandle)[tRef];
    trackClusterMap.push_back(make_pair(tRef,cRef));
  }

  // Transform Track into TransientTrack (needed by the Vertex fitter)
  edm::ESHandle<TransientTrackBuilder> theTransientTrackBuilder;
  theEventSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",theTransientTrackBuilder);

  // the output collection to be produced from this producer
  std::auto_ptr< reco::ConversionCollection > outputColl(new reco::ConversionCollection);

  // prepare iterator
  TrackClusterMap::iterator iter1    = trackClusterMap.begin();
  TrackClusterMap::iterator iter2    = trackClusterMap.begin();
  TrackClusterMap::iterator iter_end = trackClusterMap.end();

  // double-loop to make pairs
  for( ; iter1 != iter_end-1; iter1++) {
    const reco::TrackRef trk1 = iter1->first;
    const reco::CaloClusterPtr cls1 = iter1->second;
    reco::TransientTrack tsk1 = theTransientTrackBuilder->build(*trk1);
    if(!trackQualityCut(tsk1)) continue;

    for(iter2 = iter1+1; iter2 != iter_end; iter2++) {
      const reco::TrackRef trk2 = iter2->first;
      const reco::CaloClusterPtr cls2 = iter2->second;

      if(trk1 == trk2) continue;

      reco::TransientTrack tsk2 = theTransientTrackBuilder->build(*trk2);
      if(!trackQualityCut(tsk2)) continue;

      double dEta = std::abs(cls1->position().Eta() - cls2->position().Eta());
      if(dEta > 0.2) continue;
      double dPhi = std::abs(ROOT::Math::VectorUtil::DeltaPhi(cls1->position(),cls2->position()));
      if(dPhi > 0.5) continue;

      std::vector<reco::TransientTrack> toBeFitted;
      toBeFitted.push_back(tsk1);
      toBeFitted.push_back(tsk2);

      reco::Vertex theConversionVertex = (reco::Vertex) theVertexFinder_->run(toBeFitted);

      if(theConversionVertex.isValid()){
	reco::CaloClusterPtrVector scRefs;
	scRefs.push_back(cls1);
	scRefs.push_back(cls2);
	std::vector<reco::CaloClusterPtr> clusterRefs;
	clusterRefs.push_back(cls1);
	clusterRefs.push_back(cls2);
	std::vector<reco::TrackRef> trkRefs;
	trkRefs.push_back(trk1);
	trkRefs.push_back(trk2);

	std::vector<math::XYZPoint> trkPositionAtEcal = theEcalImpactPositionFinder_->find(toBeFitted,clusterBarrelHandle);
	if((clusterType_ == "BasicCluster") && (std::abs(cls2->position().Eta()) > 1.5)){
	  trkPositionAtEcal.clear();
	  trkPositionAtEcal = theEcalImpactPositionFinder_->find(toBeFitted,clusterEndcapHandle);
	}

	reco::Conversion newCandidate(scRefs, trkRefs, trkPositionAtEcal, theConversionVertex, clusterRefs);
	outputColl->push_back(newCandidate);

	printf("=====> run(%d), event(%d) <=====\n",theEvent.id().run(),theEvent.id().event());
	printf("Found a softConverion with vtxR(%f), vtxEta(%f), pt(%f), pt1(%f), pt2(%f)\n",
	       newCandidate.conversionVertex().position().rho(),newCandidate.conversionVertex().position().eta(),
	       newCandidate.pairMomentum().perp(),trk1->momentum().rho(),trk2->momentum().rho());

	clusterRefs.clear();
	trkRefs.clear();
	trkPositionAtEcal.clear();
      }// if(theConversionVertex.isValid()

      toBeFitted.clear();

    }// end of iter2
  }// end of iter1

  // put the collection into the event
  theEvent.put( outputColl, softConversionCollection_);
  
}

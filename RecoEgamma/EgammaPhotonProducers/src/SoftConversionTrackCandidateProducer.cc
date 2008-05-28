#include <iostream>
#include <vector>
#include <memory>

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
//
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/EgammaTrackReco/interface/TrackCandidateCaloClusterAssociation.h"
//
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
//
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
//  Abstract classes for the conversion tracking components
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionSeedFinder.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionTrackFinder.h"
//
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "RecoTracker/Record/interface/NavigationSchoolRecord.h"

#include "RecoEgamma/EgammaPhotonAlgos/interface/OutInConversionSeedFinder.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/InOutConversionSeedFinder.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/OutInConversionTrackFinder.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/InOutConversionTrackFinder.h"
#include "Math/GenVector/VectorUtil.h"
// Class header file
#include "RecoEgamma/EgammaPhotonProducers/interface/SoftConversionTrackCandidateProducer.h"


SoftConversionTrackCandidateProducer::SoftConversionTrackCandidateProducer(const edm::ParameterSet& config) : 
  conf_(config), 
  theNavigationSchool_(0), 
  theOutInSeedFinder_(0), 
  theOutInTrackFinder_(0), 
  theInOutSeedFinder_(0),
  theInOutTrackFinder_(0)
{
  LogDebug("SoftConversionTrackCandidateProducer") << "SoftConversionTrackCandidateProducer CTOR " << "\n";
  
  clusterType_                 = conf_.getParameter<std::string>("clusterType");
  clusterProducer_             = conf_.getParameter<std::string>("clusterProducer");
  clusterBarrelCollection_     = conf_.getParameter<std::string>("clusterBarrelCollection");
  clusterEndcapCollection_     = conf_.getParameter<std::string>("clusterEndcapCollection");
  
  OutInTrackCandidateCollection_ = conf_.getParameter<std::string>("outInTrackCandidateCollection");
  InOutTrackCandidateCollection_ = conf_.getParameter<std::string>("inOutTrackCandidateCollection");

  OutInTrackClusterAssociationCollection_ = conf_.getParameter<std::string>("outInTrackCandidateClusterAssociationCollection");
  InOutTrackClusterAssociationCollection_ = conf_.getParameter<std::string>("inOutTrackCandidateClusterAssociationCollection");

  // Register the product
  produces< TrackCandidateCollection > (OutInTrackCandidateCollection_);
  produces< TrackCandidateCollection > (InOutTrackCandidateCollection_);
  produces< reco::TrackCandidateCaloClusterPtrAssociation > ( OutInTrackClusterAssociationCollection_);
  produces< reco::TrackCandidateCaloClusterPtrAssociation > ( InOutTrackClusterAssociationCollection_);

}

SoftConversionTrackCandidateProducer::~SoftConversionTrackCandidateProducer() {
  delete theOutInSeedFinder_; 
  delete theOutInTrackFinder_;
  delete theInOutSeedFinder_;  
  delete theInOutTrackFinder_;
}

void  SoftConversionTrackCandidateProducer::setEventSetup (const edm::EventSetup & theEventSetup) {
  theOutInSeedFinder_->setEventSetup(theEventSetup);
  theInOutSeedFinder_->setEventSetup(theEventSetup);
  theOutInTrackFinder_->setEventSetup(theEventSetup);
  theInOutTrackFinder_->setEventSetup(theEventSetup);
}



void  SoftConversionTrackCandidateProducer::beginJob (edm::EventSetup const & theEventSetup) {
  nEvt_=0;
  //get magnetic field
  edm::LogInfo("SoftConversionTrackCandidateProducer") << " get magnetic field" << "\n";
  
  edm::ESHandle<NavigationSchool> nav;
  theEventSetup.get<NavigationSchoolRecord>().get("SimpleNavigationSchool", nav);
  theNavigationSchool_ = nav.product();
  
  // get the Out In Seed Finder  
  edm::LogInfo("SoftConversionTrackCandidateProducer") << " get the OutInSeedFinder" << "\n";
  theOutInSeedFinder_ = new OutInConversionSeedFinder (  conf_ );
  
  // get the Out In Track Finder
  edm::LogInfo("SoftConversionTrackCandidateProducer") << " get the OutInTrackFinder" << "\n";
  theOutInTrackFinder_ = new OutInConversionTrackFinder ( theEventSetup, conf_  );
  
  // get the In Out Seed Finder  
  edm::LogInfo("SoftConversionTrackCandidateProducer") << " get the InOutSeedFinder" << "\n";
  theInOutSeedFinder_ = new InOutConversionSeedFinder ( conf_ );
  
  // get the In Out Track Finder
  edm::LogInfo("SoftConversionTrackCandidateProducer") << " get the InOutTrackFinder" << "\n";
  theInOutTrackFinder_ = new InOutConversionTrackFinder ( theEventSetup, conf_  );
}



void SoftConversionTrackCandidateProducer::produce(edm::Event& theEvent, const edm::EventSetup& theEventSetup) {
  
  using namespace edm;
  nEvt_++;
  edm::LogInfo("SoftConversionTrackCandidateProducer") << "SoftConversionTrackCandidateProducer Analyzing event number: " << theEvent.id() << " Global Counter " << nEvt_ << "\n";
  
  setEventSetup( theEventSetup );
  theOutInSeedFinder_->setEvent(theEvent);
  theInOutSeedFinder_->setEvent(theEvent);
  theOutInTrackFinder_->setEvent(theEvent);
  theInOutTrackFinder_->setEvent(theEvent);

  // Set the navigation school  
  NavigationSetter setter(*theNavigationSchool_);  

  // collections to be stored in events
  std::auto_ptr<TrackCandidateCollection> outInTrackCandidate_p(new TrackCandidateCollection); 
  std::auto_ptr<TrackCandidateCollection> inOutTrackCandidate_p(new TrackCandidateCollection); 
  std::auto_ptr<reco::TrackCandidateCaloClusterPtrAssociation> outInAssoc_p(new reco::TrackCandidateCaloClusterPtrAssociation);
  std::auto_ptr<reco::TrackCandidateCaloClusterPtrAssociation> inOutAssoc_p(new reco::TrackCandidateCaloClusterPtrAssociation);
   
  std::vector<edm::Ptr<reco::CaloCluster> > vecOfClusterRefForOutIn;
  std::vector<edm::Ptr<reco::CaloCluster> > vecOfClusterRefForInOut;

  // Get the basic cluster collection in the Barrel 
  edm::Handle<edm::View<reco::CaloCluster> > clusterBarrelHandle;
  theEvent.getByLabel(clusterProducer_, clusterBarrelCollection_, clusterBarrelHandle);
  if (!clusterBarrelHandle.isValid()) {
    edm::LogError("SoftConverionTrackCandidateProducer") << "Error! Can't get the product "<<clusterBarrelCollection_.c_str();
    return;
  }
  
  buildCollections(clusterBarrelHandle, *outInTrackCandidate_p, *inOutTrackCandidate_p, vecOfClusterRefForOutIn, vecOfClusterRefForInOut);

  if(clusterType_ == "BasicCluster") {
    // Get the basic cluster collection in the Endcap 
    edm::Handle<edm::View<reco::CaloCluster> > clusterEndcapHandle;
    theEvent.getByLabel(clusterProducer_, clusterEndcapCollection_, clusterEndcapHandle);
    if (!clusterEndcapHandle.isValid()) {
      edm::LogError("SoftConversionTrackCandidateProducer") << "Error! Can't get the product "<<clusterEndcapCollection_.c_str();
      return;
    }

    buildCollections(clusterEndcapHandle, *outInTrackCandidate_p, *inOutTrackCandidate_p, vecOfClusterRefForOutIn, vecOfClusterRefForInOut);

  }

  // put all products in the event
  const edm::OrphanHandle<TrackCandidateCollection> refprodOutInTrackC = theEvent.put( outInTrackCandidate_p, OutInTrackCandidateCollection_ );
  const edm::OrphanHandle<TrackCandidateCollection> refprodInOutTrackC = theEvent.put( inOutTrackCandidate_p, InOutTrackCandidateCollection_ );

  edm::ValueMap<reco::CaloClusterPtr>::Filler fillerOI(*outInAssoc_p);
  fillerOI.insert(refprodOutInTrackC, vecOfClusterRefForOutIn.begin(), vecOfClusterRefForOutIn.end());
  fillerOI.fill();
  edm::ValueMap<reco::CaloClusterPtr>::Filler fillerIO(*inOutAssoc_p);
  fillerIO.insert(refprodInOutTrackC, vecOfClusterRefForInOut.begin(), vecOfClusterRefForInOut.end());
  fillerIO.fill();

  theEvent.put( outInAssoc_p, OutInTrackClusterAssociationCollection_);
  theEvent.put( inOutAssoc_p, InOutTrackClusterAssociationCollection_);
}


void SoftConversionTrackCandidateProducer::buildCollections(const edm::Handle<edm::View<reco::CaloCluster> > & clusterHandle,
							    TrackCandidateCollection& outInTrackCandidates,
							    TrackCandidateCollection& inOutTrackCandidates,
							    std::vector<edm::Ptr<reco::CaloCluster> >& vecRecOI,
							    std::vector<edm::Ptr<reco::CaloCluster> >& vecRecIO) {

  int nClusters = (int) clusterHandle->size();
  for(int iCluster=0; iCluster<nClusters; iCluster++){
    reco::CaloClusterPtr clusterRefOutIn = clusterHandle->ptrAt(iCluster);
    math::XYZPoint position = clusterRefOutIn->position();
    GlobalPoint gp(position.x(),position.y(),position.z());
    theOutInSeedFinder_->setCandidate(clusterRefOutIn->energy(),gp);
    theOutInSeedFinder_->makeSeeds(clusterRefOutIn);

    std::vector<Trajectory> theOutInTracks= theOutInTrackFinder_->tracks(theOutInSeedFinder_->seeds(), outInTrackCandidates);

    int nOITrj = (int) theOutInTracks.size();
    for(int itrj=0; itrj < nOITrj; itrj++) vecRecOI.push_back( clusterRefOutIn );

    for(int jCluster=iCluster; jCluster<nClusters; jCluster++){
      reco::CaloClusterPtr clusterRefInOut = clusterHandle->ptrAt(jCluster);

      math::XYZPoint position2 = clusterRefInOut->position();
      GlobalPoint gp2(position2.x(),position2.y(),position2.z());
      double dEta = std::abs(position.Eta() - position2.Eta());
      if(dEta > 0.2) continue;

      double dPhi = std::abs(ROOT::Math::VectorUtil::DeltaPhi(position, position2));
      if(dPhi > 0.5) continue;

      theInOutSeedFinder_->setCandidate(clusterRefInOut->energy(),gp2);
      theInOutSeedFinder_->setTracks(theOutInTracks);   
      theInOutSeedFinder_->makeSeeds(clusterHandle);
    
      std::vector<Trajectory> theInOutTracks= theInOutTrackFinder_->tracks(theInOutSeedFinder_->seeds(), inOutTrackCandidates); 

      int nIOTrj = (int) theInOutTracks.size();
      for(int itrj=0; itrj < nIOTrj; itrj++) vecRecIO.push_back( clusterRefInOut );

    }// for jCluster
  }// for iCluster

}


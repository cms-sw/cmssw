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

bool IsGoodSeed(const TrajectorySeedCollection& seeds, const TrajectorySeed& seed){

  // This function is not satisfactory. I don't know how to check equality of TrajectorySeed
  // So I compare all possible quantities in them.
  // This function can be dropped when I find to check equality of TrajectorySeed.

  bool found = false;
  for(TrajectorySeedCollection::const_iterator it = seeds.begin(); it != seeds.end(); it++){
    if(it->nHits() != seed.nHits()) continue;
    if(it->startingState().detId() != seed.startingState().detId()) continue;
    if(it->startingState().surfaceSide() != seed.startingState().surfaceSide()) continue;
    if((it->startingState().parameters().position() - seed.startingState().parameters().position()).mag() > 1.0e-6) continue;
    if((it->startingState().parameters().momentum() - seed.startingState().parameters().momentum()).mag() > 1.0e-6) continue;
    found = true;
    break;
  }

  return found;
}

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
  clusterBarrelCollection_     = conf_.getParameter<edm::InputTag>("clusterBarrelCollection");
  clusterEndcapCollection_     = conf_.getParameter<edm::InputTag>("clusterEndcapCollection");
  
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

SoftConversionTrackCandidateProducer::~SoftConversionTrackCandidateProducer() {}

void  SoftConversionTrackCandidateProducer::setEventSetup (const edm::EventSetup & theEventSetup) {
  theOutInSeedFinder_->setEventSetup(theEventSetup);
  theInOutSeedFinder_->setEventSetup(theEventSetup);
  theOutInTrackFinder_->setEventSetup(theEventSetup);
  theInOutTrackFinder_->setEventSetup(theEventSetup);
}



void  SoftConversionTrackCandidateProducer::beginRun (edm::Run& r, edm::EventSetup const & theEventSetup) {
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

void  SoftConversionTrackCandidateProducer::endRun (edm::Run& r, edm::EventSetup const & theEventSetup) {
  delete theOutInSeedFinder_; 
  delete theOutInTrackFinder_;
  delete theInOutSeedFinder_;  
  delete theInOutTrackFinder_;
}


void SoftConversionTrackCandidateProducer::produce(edm::Event& theEvent, const edm::EventSetup& theEventSetup) {
  
  using namespace edm;
  nEvt_++;
  edm::LogInfo("SoftConversionTrackCandidateProducer") << "SoftConversionTrackCandidateProducer Analyzing event number: " << theEvent.id() << " Global Counter " << nEvt_ << "\n";

  std::cout << "SoftConversionTrackCandidateProducer Analyzing event number: " << theEvent.id() << " Global Counter " << nEvt_ << "\n";
  
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
  theEvent.getByLabel(clusterBarrelCollection_, clusterBarrelHandle);

  if (!clusterBarrelHandle.isValid()) {
    edm::LogError("SoftConverionTrackCandidateProducer") << "Error! Can't get the product "<<clusterBarrelCollection_.label();
    return;
  }
  
  buildCollections(clusterBarrelHandle, *outInTrackCandidate_p, *inOutTrackCandidate_p, vecOfClusterRefForOutIn, vecOfClusterRefForInOut);

  if(clusterType_ == "BasicCluster" ) {
    // Get the basic cluster collection in the Endcap 
    edm::Handle<edm::View<reco::CaloCluster> > clusterEndcapHandle;
    theEvent.getByLabel(clusterEndcapCollection_, clusterEndcapHandle);

    if (!clusterEndcapHandle.isValid()) {
      edm::LogError("SoftConversionTrackCandidateProducer") << "Error! Can't get the product "<<clusterEndcapCollection_.label();
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

  // temporary collection
  TrackCandidateCollection tempTCC;
  TrajectorySeedCollection totalOISeeds; // total number of out-in trajectory seeds through entire cluster collection loop
  TrajectorySeedCollection totalIOSeeds; // total number of in-out trajectory seeds through entire cluster collection loop

  int nClusters = (int) clusterHandle->size();

  // first loop to fill totalOISeeds and totalIOSeeds

  for(int iCluster=0; iCluster<nClusters; iCluster++){
    reco::CaloClusterPtr clusterRefOutIn = clusterHandle->ptrAt(iCluster);
    math::XYZPoint position = clusterRefOutIn->position();
    GlobalPoint gp(position.x(),position.y(),position.z());
    theOutInSeedFinder_->setCandidate(clusterRefOutIn->energy(),gp);
    theOutInSeedFinder_->makeSeeds(clusterRefOutIn);


    TrajectorySeedCollection oISeeds = theOutInSeedFinder_->seeds();
    for(TrajectorySeedCollection::const_iterator it = oISeeds.begin(); it != oISeeds.end(); it++){
      totalOISeeds.push_back(*it);
    }

    std::vector<Trajectory> theOutInTracks= theOutInTrackFinder_->tracks(oISeeds, tempTCC);
    tempTCC.clear();

    for(int jCluster=iCluster; jCluster<nClusters; jCluster++){
      reco::CaloClusterPtr clusterRefInOut = clusterHandle->ptrAt(jCluster);

      math::XYZPoint position2 = clusterRefInOut->position();
      GlobalPoint gp2(position2.x(),position2.y(),position2.z());
      double dEta = std::abs(position.Eta() - position2.Eta());
      if(dEta > 0.1) continue;

      double dPhi = std::abs(ROOT::Math::VectorUtil::DeltaPhi(position, position2));
      if(dPhi > 0.5) continue;

      theInOutSeedFinder_->setCandidate(clusterRefInOut->energy(),gp2);
      theInOutSeedFinder_->setTracks(theOutInTracks);   
      theInOutSeedFinder_->makeSeeds(clusterHandle);

      TrajectorySeedCollection iOSeeds = theInOutSeedFinder_->seeds();

      for(TrajectorySeedCollection::const_iterator it = iOSeeds.begin(); it != iOSeeds.end(); it++){
	totalIOSeeds.push_back(*it);
      }
    
    }// for jCluster
  }// for iCluster


  // Now we have total OI/IO seeds. Let's clean them up and save them with only giving good trajectories
  TrajectorySeedCollection oIFilteredSeeds;
  TrajectorySeedCollection iOFilteredSeeds;

  tempTCC.clear();
  std::vector<Trajectory> tempTrj = theOutInTrackFinder_->tracks(totalOISeeds,tempTCC);
  for(std::vector<Trajectory>::iterator it = tempTrj.begin(); it!= tempTrj.end(); it++){
    oIFilteredSeeds.push_back(it->seed());
  }

  tempTrj.clear();
  tempTCC.clear();
  tempTrj = theInOutTrackFinder_->tracks(totalIOSeeds,tempTCC);
  for(std::vector<Trajectory>::iterator it = tempTrj.begin(); it!= tempTrj.end(); it++){
    iOFilteredSeeds.push_back(it->seed());
  }

  tempTCC.clear();
  tempTrj.clear();
  totalOISeeds.clear();
  totalIOSeeds.clear();


  // Now start normal procedure and consider seeds that belong to filtered ones.

  for(int iCluster=0; iCluster<nClusters; iCluster++){
    reco::CaloClusterPtr clusterRefOutIn = clusterHandle->ptrAt(iCluster);
    math::XYZPoint position = clusterRefOutIn->position();
    GlobalPoint gp(position.x(),position.y(),position.z());
    theOutInSeedFinder_->setCandidate(clusterRefOutIn->energy(),gp);
    theOutInSeedFinder_->makeSeeds(clusterRefOutIn);

    TrajectorySeedCollection oISeeds_all = theOutInSeedFinder_->seeds();
    TrajectorySeedCollection oISeeds;
    for(TrajectorySeedCollection::iterator it = oISeeds_all.begin(); it != oISeeds_all.end(); it++){
      if(IsGoodSeed(oIFilteredSeeds,*it)) oISeeds.push_back(*it);
    }

    if(oISeeds.size() == 0) continue;

    std::vector<Trajectory> theOutInTracks= theOutInTrackFinder_->tracks(oISeeds, outInTrackCandidates);

    int nOITrj = (int) theOutInTracks.size();
    for(int itrj=0; itrj < nOITrj; itrj++) vecRecOI.push_back( clusterRefOutIn );

    for(int jCluster=iCluster; jCluster<nClusters; jCluster++){
      reco::CaloClusterPtr clusterRefInOut = clusterHandle->ptrAt(jCluster);

      math::XYZPoint position2 = clusterRefInOut->position();
      GlobalPoint gp2(position2.x(),position2.y(),position2.z());
      double dEta = std::abs(position.Eta() - position2.Eta());
      if(dEta > 0.1) continue;

      double dPhi = std::abs(ROOT::Math::VectorUtil::DeltaPhi(position, position2));
      if(dPhi > 0.5) continue;

      theInOutSeedFinder_->setCandidate(clusterRefInOut->energy(),gp2);
      theInOutSeedFinder_->setTracks(theOutInTracks);   
      theInOutSeedFinder_->makeSeeds(clusterHandle);
    
      TrajectorySeedCollection iOSeeds_all = theInOutSeedFinder_->seeds();
      TrajectorySeedCollection iOSeeds;
      for(TrajectorySeedCollection::iterator it = iOSeeds_all.begin(); it != iOSeeds_all.end(); it++){
	if(IsGoodSeed(iOFilteredSeeds,*it)) iOSeeds.push_back(*it);
      }

      if(iOSeeds.size() == 0) continue;

      std::vector<Trajectory> theInOutTracks= theInOutTrackFinder_->tracks(iOSeeds, inOutTrackCandidates); 

      int nIOTrj = (int) theInOutTracks.size();
      for(int itrj=0; itrj < nIOTrj; itrj++) vecRecIO.push_back( clusterRefInOut );

    }// for jCluster
  }// for iCluster


}


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
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/EgammaTrackReco/interface/TrackCandidateSuperClusterAssociation.h"
#include "DataFormats/EgammaTrackReco/interface/TrackCandidateCaloClusterAssociation.h"
//
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
//
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
//  Abstract classes for the conversion tracking components
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionSeedFinder.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionTrackFinder.h"
// Class header file
#include "RecoEgamma/EgammaPhotonProducers/interface/ConversionTrackCandidateProducer.h"
//
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "RecoTracker/Record/interface/NavigationSchoolRecord.h"

#include "RecoEgamma/EgammaPhotonAlgos/interface/OutInConversionSeedFinder.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/InOutConversionSeedFinder.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/OutInConversionTrackFinder.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/InOutConversionTrackFinder.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"

ConversionTrackCandidateProducer::ConversionTrackCandidateProducer(const edm::ParameterSet& config) : 
  conf_(config), 
  theNavigationSchool_(0), 
  theOutInSeedFinder_(0), 
  theOutInTrackFinder_(0), 
  theInOutSeedFinder_(0),
  theInOutTrackFinder_(0)
{


  
  //std::cout << "ConversionTrackCandidateProducer CTOR " << "\n";
  
   
  // use onfiguration file to setup input/output collection names
 
//Using InputTag
//  bcProducer_             = conf_.getParameter<std::string>("bcProducer");
  bcBarrelCollection_     = conf_.getParameter<edm::InputTag>("bcBarrelCollection");
  bcEndcapCollection_     = conf_.getParameter<edm::InputTag>("bcEndcapCollection");
  
  scHybridBarrelProducer_       = conf_.getParameter<std::string>("scHybridBarrelProducer");
  scIslandEndcapProducer_       = conf_.getParameter<std::string>("scIslandEndcapProducer");
  
  scHybridBarrelCollection_     = conf_.getParameter<std::string>("scHybridBarrelCollection");
  scIslandEndcapCollection_     = conf_.getParameter<std::string>("scIslandEndcapCollection");
  
  OutInTrackCandidateCollection_ = conf_.getParameter<std::string>("outInTrackCandidateCollection");
  InOutTrackCandidateCollection_ = conf_.getParameter<std::string>("inOutTrackCandidateCollection");


  OutInTrackSuperClusterAssociationCollection_ = conf_.getParameter<std::string>("outInTrackCandidateSCAssociationCollection");
  InOutTrackSuperClusterAssociationCollection_ = conf_.getParameter<std::string>("inOutTrackCandidateSCAssociationCollection");

  hbheLabel_        = conf_.getParameter<std::string>("hbheModule");
  hbheInstanceName_ = conf_.getParameter<std::string>("hbheInstance");
  hOverEConeSize_   = conf_.getParameter<double>("hOverEConeSize");
  maxHOverE_        = conf_.getParameter<double>("maxHOverE");
  minSCEt_        = conf_.getParameter<double>("minSCEt");


  // Register the product
  produces< TrackCandidateCollection > (OutInTrackCandidateCollection_);
  produces< TrackCandidateCollection > (InOutTrackCandidateCollection_);

  produces< reco::TrackCandidateCaloClusterPtrAssociation > ( OutInTrackSuperClusterAssociationCollection_);
  produces< reco::TrackCandidateCaloClusterPtrAssociation > ( InOutTrackSuperClusterAssociationCollection_);
  

}

ConversionTrackCandidateProducer::~ConversionTrackCandidateProducer() {


  delete theOutInSeedFinder_; 
  delete theOutInTrackFinder_;
  delete theInOutSeedFinder_;  
  delete theInOutTrackFinder_;


}

void  ConversionTrackCandidateProducer::setEventSetup (const edm::EventSetup & theEventSetup) {


  theOutInSeedFinder_->setEventSetup(theEventSetup);
  theInOutSeedFinder_->setEventSetup(theEventSetup);
  theOutInTrackFinder_->setEventSetup(theEventSetup);
  theInOutTrackFinder_->setEventSetup(theEventSetup);

}



void  ConversionTrackCandidateProducer::beginJob (edm::EventSetup const & theEventSetup) {
  nEvt_=0;
  //get magnetic field
  edm::LogInfo("ConversionTrackCandidateProducer") << " get magnetic field" << "\n";
  
  edm::ESHandle<NavigationSchool> nav;
  theEventSetup.get<NavigationSchoolRecord>().get("SimpleNavigationSchool", nav);
  theNavigationSchool_ = nav.product();
  
  // get the Out In Seed Finder  
  edm::LogInfo("ConversionTrackCandidateProducer") << " get the OutInSeedFinder" << "\n";
  theOutInSeedFinder_ = new OutInConversionSeedFinder (  conf_ );
  
  // get the Out In Track Finder
  edm::LogInfo("ConversionTrackCandidateProducer") << " get the OutInTrackFinder" << "\n";
  theOutInTrackFinder_ = new OutInConversionTrackFinder ( theEventSetup, conf_  );

  
  // get the In Out Seed Finder  
  edm::LogInfo("ConversionTrackCandidateProducer") << " get the InOutSeedFinder" << "\n";
  theInOutSeedFinder_ = new InOutConversionSeedFinder ( conf_ );
  
  
  // get the In Out Track Finder
  edm::LogInfo("ConversionTrackCandidateProducer") << " get the InOutTrackFinder" << "\n";
  theInOutTrackFinder_ = new InOutConversionTrackFinder ( theEventSetup, conf_  );
  
  
}



void ConversionTrackCandidateProducer::produce(edm::Event& theEvent, const edm::EventSetup& theEventSetup) {
  
  using namespace edm;
  nEvt_++;
  edm::LogInfo("ConversionTrackCandidateProducer") << "ConversionTrackCandidateProducer Analyzing event number: " << theEvent.id() << " Global Counter " << nEvt_ << "\n";
  //  std::cout << "ConversionTrackCandidateProducer Analyzing event number " <<   theEvent.id() <<  " Global Counter " << nEvt_ << "\n";
  

  
  setEventSetup( theEventSetup );
  theOutInSeedFinder_->setEvent(theEvent);
  theInOutSeedFinder_->setEvent(theEvent);
  theOutInTrackFinder_->setEvent(theEvent);
  theInOutTrackFinder_->setEvent(theEvent);

// Set the navigation school  
  NavigationSetter setter(*theNavigationSchool_);  

  //
  // create empty output collections
  //
  //  Out In Track Candidates
  std::auto_ptr<TrackCandidateCollection> outInTrackCandidate_p(new TrackCandidateCollection); 
  //  In Out  Track Candidates
  std::auto_ptr<TrackCandidateCollection> inOutTrackCandidate_p(new TrackCandidateCollection); 
  //   Track Candidate  calo  Cluster Association
  std::auto_ptr<reco::TrackCandidateCaloClusterPtrAssociation> outInAssoc_p(new reco::TrackCandidateCaloClusterPtrAssociation);
  std::auto_ptr<reco::TrackCandidateCaloClusterPtrAssociation> inOutAssoc_p(new reco::TrackCandidateCaloClusterPtrAssociation);
    
  // Get the basic cluster collection in the Barrel 
  edm::Handle<edm::View<reco::CaloCluster> > bcBarrelHandle;
  theEvent.getByLabel(bcBarrelCollection_, bcBarrelHandle);
  if (!bcBarrelHandle.isValid()) {
    edm::LogError("ConversionTrackCandidateProducer") << "Error! Can't get the product "<<bcBarrelCollection_;
    return;
  }
  
  
  // Get the basic cluster collection in the Endcap 
  edm::Handle<edm::View<reco::CaloCluster> > bcEndcapHandle;
  theEvent.getByLabel(bcEndcapCollection_, bcEndcapHandle);
  if (!bcEndcapHandle.isValid()) {
    edm::LogError("CoonversionTrackCandidateProducer") << "Error! Can't get the product "<<bcEndcapCollection_;
    return;
  }
  
  

  // Get the Super Cluster collection in the Barrel
  edm::Handle<edm::View<reco::CaloCluster> > scBarrelHandle;
  theEvent.getByLabel(scHybridBarrelProducer_,scHybridBarrelCollection_,scBarrelHandle);
  if (!scBarrelHandle.isValid()) {
    edm::LogError("CoonversionTrackCandidateProducer") << "Error! Can't get the product "<<scHybridBarrelCollection_.c_str();
    return;
  }


  // Get the Super Cluster collection in the Endcap
  edm::Handle<edm::View<reco::CaloCluster> > scEndcapHandle;
  theEvent.getByLabel(scIslandEndcapProducer_,scIslandEndcapCollection_,scEndcapHandle);
  if (!scEndcapHandle.isValid()) {
    edm::LogError("CoonversionTrackCandidateProducer") << "Error! Can't get the product "<<scIslandEndcapCollection_.c_str();
    return;
  }


  // get the geometry from the event setup:
  theEventSetup.get<CaloGeometryRecord>().get(theCaloGeom_);
  // Get HoverE
  Handle<HBHERecHitCollection> hbhe;
  std::auto_ptr<HBHERecHitMetaCollection> mhbhe;
  theEvent.getByLabel(hbheLabel_,hbheInstanceName_,hbhe);  
  if (!hbhe.isValid()) {
    edm::LogError("PhotonProducer") << "Error! Can't get the product "<<hbheInstanceName_.c_str();
    return; 
  }

  
  if (hOverEConeSize_ > 0.) {
    mhbhe=  std::auto_ptr<HBHERecHitMetaCollection>(new HBHERecHitMetaCollection(*hbhe));
  }
  theHoverEcalc_=HoECalculator(theCaloGeom_);


  caloPtrVecOutIn_.clear();
  caloPtrVecInOut_.clear();

  buildCollections(scBarrelHandle, bcBarrelHandle,  mhbhe.get(), *outInTrackCandidate_p,*inOutTrackCandidate_p,caloPtrVecOutIn_,caloPtrVecInOut_ );
  buildCollections(scEndcapHandle, bcEndcapHandle,  mhbhe.get(), *outInTrackCandidate_p,*inOutTrackCandidate_p,caloPtrVecOutIn_,caloPtrVecInOut_ );




  //  std::cout  << "  ConversionTrackCandidateProducer  caloPtrVecOutIn_ size " <<  caloPtrVecOutIn_.size() << " caloPtrVecInOut_ size " << caloPtrVecInOut_.size()  << "\n"; 
  


  // put all products in the event
 // Barrel
 //std::cout  << "ConversionTrackCandidateProducer Putting in the event " << (*outInTrackCandidate_p).size() << " Out In track Candidates " << "\n";
 edm::LogInfo("ConversionTrackCandidateProducer") << "Number of outInTrackCandidates: " <<  (*outInTrackCandidate_p).size() << "\n";
 const edm::OrphanHandle<TrackCandidateCollection> refprodOutInTrackC = theEvent.put( outInTrackCandidate_p, OutInTrackCandidateCollection_ );
 //std::cout  << "ConversionTrackCandidateProducer  refprodOutInTrackC size  " <<  (*(refprodOutInTrackC.product())).size()  <<  "\n";
 //
 //std::cout  << "ConversionTrackCandidateProducer Putting in the event  " << (*inOutTrackCandidate_p).size() << " In Out track Candidates " <<  "\n";
 edm::LogInfo("ConversionTrackCandidateProducer") << "Number of inOutTrackCandidates: " <<  (*inOutTrackCandidate_p).size() << "\n";
 const edm::OrphanHandle<TrackCandidateCollection> refprodInOutTrackC = theEvent.put( inOutTrackCandidate_p, InOutTrackCandidateCollection_ );
 //std::cout  << "ConversionTrackCandidateProducer  refprodInOutTrackC size  " <<  (*(refprodInOutTrackC.product())).size()  <<  "\n";


 edm::ValueMap<reco::CaloClusterPtr>::Filler fillerOI(*outInAssoc_p);
 fillerOI.insert(refprodOutInTrackC, caloPtrVecOutIn_.begin(), caloPtrVecOutIn_.end());
 fillerOI.fill();
 edm::ValueMap<reco::CaloClusterPtr>::Filler fillerIO(*inOutAssoc_p);
 fillerIO.insert(refprodInOutTrackC, caloPtrVecInOut_.begin(), caloPtrVecInOut_.end());
 fillerIO.fill();


  
 // std::cout  << "ConversionTrackCandidateProducer Putting in the event   OutIn track - SC association: size  " <<  (*outInAssoc_p).size() << "\n";  
 theEvent.put( outInAssoc_p, OutInTrackSuperClusterAssociationCollection_);
 
 // std::cout << "ConversionTrackCandidateProducer Putting in the event   InOut track - SC association: size  " <<  (*inOutAssoc_p).size() << "\n";  
 theEvent.put( inOutAssoc_p, InOutTrackSuperClusterAssociationCollection_);



  
}


void ConversionTrackCandidateProducer::buildCollections( const edm::Handle<edm::View<reco::CaloCluster> > & scHandle,
                                                         const edm::Handle<edm::View<reco::CaloCluster> > & bcHandle,
							 HBHERecHitMetaCollection *mhbhe,
							 TrackCandidateCollection& outInTrackCandidates,
							 TrackCandidateCollection& inOutTrackCandidates,
							 std::vector<edm::Ptr<reco::CaloCluster> >& vecRecOI,
							 std::vector<edm::Ptr<reco::CaloCluster> >& vecRecIO )

{

  //  std::cout << "ConversionTrackCandidateProducer builcollections bc size " << bcHandle->size() <<  "\n";

  //  Loop over SC in the barrel and reconstruct converted photons
  for (unsigned i = 0; i < scHandle->size(); ++i ) {

    reco::CaloClusterPtr aClus= scHandle->ptrAt(i);
    // preselection
    if (aClus->energy()/cosh(aClus->eta()) <= minSCEt_) continue;


    const reco::CaloCluster* pClus=&(*aClus);
    const reco::SuperCluster*  sc=dynamic_cast<const reco::SuperCluster*>(pClus);
    double HoE=theHoverEcalc_(sc,mhbhe);
    if (HoE>=maxHOverE_)  continue;

    theOutInSeedFinder_->setCandidate(pClus->energy(), GlobalPoint(pClus->position().x(),pClus->position().y(),pClus->position().z() ) );
    theOutInSeedFinder_->makeSeeds( bcHandle );

    std::vector<Trajectory> theOutInTracks= theOutInTrackFinder_->tracks(theOutInSeedFinder_->seeds(),  outInTrackCandidates);    

 
    theInOutSeedFinder_->setCandidate(pClus->energy(), GlobalPoint(pClus->position().x(),pClus->position().y(),pClus->position().z() ) );  
    theInOutSeedFinder_->setTracks(  theOutInTracks );   
    theInOutSeedFinder_->makeSeeds(  bcHandle);
    
    std::vector<Trajectory> theInOutTracks= theInOutTrackFinder_->tracks(theInOutSeedFinder_->seeds(),  inOutTrackCandidates); 


    // Debug
    //   std::cout  << "ConversionTrackCandidateProducer  theOutInTracks.size() " << theOutInTracks.size() << " theInOutTracks.size() " << theInOutTracks.size() <<  " Event pointer to out in track size barrel " << outInTrackCandidates.size() << " in out track size " << inOutTrackCandidates.size() <<   "\n";


   //////////// Fill vectors of Ref to SC to be used for the Track-SC association
    for (std::vector<Trajectory>::const_iterator it = theOutInTracks.begin(); it !=  theOutInTracks.end(); ++it) {
      caloPtrVecOutIn_.push_back(aClus);
      //     std::cout  << "ConversionTrackCandidateProducer Barrel OutIn Tracks Number of hits " << (*it).foundHits() << "\n"; 
    }

    for (std::vector<Trajectory>::const_iterator it = theInOutTracks.begin(); it !=  theInOutTracks.end(); ++it) {
      caloPtrVecInOut_.push_back(aClus);
      //     std::cout  << "ConversionTrackCandidateProducer Barrel InOut Tracks Number of hits " << (*it).foundHits() << "\n"; 
    }




    

  }



}


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
#include "FWCore/Framework/interface/ConsumesCollector.h"
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
#include "RecoTracker/CkfPattern/interface/BaseCkfTrajectoryBuilderFactory.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "RecoTracker/Record/interface/NavigationSchoolRecord.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/OutInConversionSeedFinder.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/InOutConversionSeedFinder.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/OutInConversionTrackFinder.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/InOutConversionTrackFinder.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaTowerIsolation.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaRecHitIsolation.h"

#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "CommonTools/Utils/interface/StringToEnumValue.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"

#include "RecoTracker/CkfPattern/interface/BaseCkfTrajectoryBuilder.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"

namespace {
  BaseCkfTrajectoryBuilder *createBaseCkfTrajectoryBuilder(const edm::ParameterSet& pset, edm::ConsumesCollector&& iC) {
    return BaseCkfTrajectoryBuilderFactory::get()->create(pset.getParameter<std::string>("ComponentType"), pset, iC);
  }
}

ConversionTrackCandidateProducer::ConversionTrackCandidateProducer(const edm::ParameterSet& config) : 
  theTrajectoryBuilder_(createBaseCkfTrajectoryBuilder(config.getParameter<edm::ParameterSet>("TrajectoryBuilderPSet"), consumesCollector())),
  theNavigationSchool_(0), 
  theOutInSeedFinder_(new OutInConversionSeedFinder(config)),
  theOutInTrackFinder_(new OutInConversionTrackFinder(config, theTrajectoryBuilder_.get())),
  theInOutSeedFinder_(new InOutConversionSeedFinder(config)),
  theInOutTrackFinder_(new InOutConversionTrackFinder(config, theTrajectoryBuilder_.get()))
{  
  //std::cout << "ConversionTrackCandidateProducer CTOR " << "\n";
  nEvt_=0;  
   
  // use onfiguration file to setup input/output collection names
 

  bcBarrelCollection_ = 
    consumes<edm::View<reco::CaloCluster> >(config.getParameter<edm::InputTag>("bcBarrelCollection"));
  bcEndcapCollection_ = 
    consumes<edm::View<reco::CaloCluster> >(config.getParameter<edm::InputTag>("bcEndcapCollection"));
  
  scHybridBarrelProducer_ = 
    consumes<edm::View<reco::CaloCluster> >(config.getParameter<edm::InputTag>("scHybridBarrelProducer"));
  scIslandEndcapProducer_ = 
    consumes<edm::View<reco::CaloCluster> >(config.getParameter<edm::InputTag>("scIslandEndcapProducer"));
  
  OutInTrackCandidateCollection_ = config.getParameter<std::string>("outInTrackCandidateCollection");
  InOutTrackCandidateCollection_ = config.getParameter<std::string>("inOutTrackCandidateCollection");


  OutInTrackSuperClusterAssociationCollection_ = config.getParameter<std::string>("outInTrackCandidateSCAssociationCollection");
  InOutTrackSuperClusterAssociationCollection_ = config.getParameter<std::string>("inOutTrackCandidateSCAssociationCollection");

  barrelecalCollection_ = 
    consumes<EcalRecHitCollection>(config.getParameter<edm::InputTag>("barrelEcalRecHitCollection"));
  endcapecalCollection_ = 
    consumes<EcalRecHitCollection>(config.getParameter<edm::InputTag>("endcapEcalRecHitCollection"));
  
  hcalTowers_        = 
    consumes<CaloTowerCollection>(config.getParameter<edm::InputTag>("hcalTowers"));
  hOverEConeSize_    = config.getParameter<double>("hOverEConeSize");
  maxHOverE_         = config.getParameter<double>("maxHOverE");
  minSCEt_           = config.getParameter<double>("minSCEt");
  isoConeR_          = config.getParameter<double>("isoConeR");
  isoInnerConeR_     = config.getParameter<double>("isoInnerConeR");
  isoEtaSlice_       = config.getParameter<double>("isoEtaSlice");
  isoEtMin_          = config.getParameter<double>("isoEtMin");
  isoEMin_           = config.getParameter<double>("isoEMin");
  vetoClusteredHits_ = config.getParameter<bool>("vetoClusteredHits");
  useNumXtals_       = config.getParameter<bool>("useNumXstals");
   ecalIsoCut_offset_ = config.getParameter<double>("ecalIsoCut_offset");
  ecalIsoCut_slope_  = config.getParameter<double>("ecalIsoCut_slope");

  //Flags and Severities to be excluded from photon calculations
  const std::vector<std::string> flagnamesEB = 
    config.getParameter<std::vector<std::string> >("RecHitFlagToBeExcludedEB");

  const std::vector<std::string> flagnamesEE =
    config.getParameter<std::vector<std::string> >("RecHitFlagToBeExcludedEE");

  flagsexclEB_= 
    StringToEnumValue<EcalRecHit::Flags>(flagnamesEB);

  flagsexclEE_=
    StringToEnumValue<EcalRecHit::Flags>(flagnamesEE);

  const std::vector<std::string> severitynamesEB = 
    config.getParameter<std::vector<std::string> >("RecHitSeverityToBeExcludedEB");

  severitiesexclEB_= 
    StringToEnumValue<EcalSeverityLevel::SeverityLevel>(severitynamesEB);

  const std::vector<std::string> severitynamesEE = 
    config.getParameter<std::vector<std::string> >("RecHitSeverityToBeExcludedEE");

  severitiesexclEE_= 
    StringToEnumValue<EcalSeverityLevel::SeverityLevel>(severitynamesEE);

  // Register the product
  produces< TrackCandidateCollection > (OutInTrackCandidateCollection_);
  produces< TrackCandidateCollection > (InOutTrackCandidateCollection_);

  produces< reco::TrackCandidateCaloClusterPtrAssociation > ( OutInTrackSuperClusterAssociationCollection_);
  produces< reco::TrackCandidateCaloClusterPtrAssociation > ( InOutTrackSuperClusterAssociationCollection_);
  

}

ConversionTrackCandidateProducer::~ConversionTrackCandidateProducer() {}

void  ConversionTrackCandidateProducer::setEventSetup (const edm::EventSetup & theEventSetup) {
  theOutInSeedFinder_->setEventSetup(theEventSetup);
  theInOutSeedFinder_->setEventSetup(theEventSetup);
  theOutInTrackFinder_->setEventSetup(theEventSetup);
  theInOutTrackFinder_->setEventSetup(theEventSetup);
}


void  ConversionTrackCandidateProducer::beginRun (edm::Run const& r , edm::EventSetup const & theEventSetup) {

  edm::ESHandle<NavigationSchool> nav;
  theEventSetup.get<NavigationSchoolRecord>().get("SimpleNavigationSchool", nav);
  theNavigationSchool_ = nav.product();
}


void ConversionTrackCandidateProducer::produce(edm::Event& theEvent, const edm::EventSetup& theEventSetup) {
  
  using namespace edm;
  nEvt_++;
  //  std::cout << "ConversionTrackCandidateProducer Analyzing event number " <<   theEvent.id() <<  " Global Counter " << nEvt_ << "\n";
  

  
  // get the trajectory builder and initialize it with the data
  edm::Handle<MeasurementTrackerEvent> data;
  theEvent.getByLabel(edm::InputTag("MeasurementTrackerEvent"), data);
  theTrajectoryBuilder_->setEvent(theEvent, theEventSetup, &*data);


  // this need to be done after the initialization of the TrajectoryBuilder!
  setEventSetup( theEventSetup );
 


  theOutInSeedFinder_->setEvent(theEvent);
  theInOutSeedFinder_->setEvent(theEvent);

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
  bool validBarrelBCHandle=true;
  edm::Handle<edm::View<reco::CaloCluster> > bcBarrelHandle;
  theEvent.getByToken(bcBarrelCollection_, bcBarrelHandle);
  if (!bcBarrelHandle.isValid()) {
    edm::LogError("ConversionTrackCandidateProducer") 
      << "Error! Can't get the Barrel Basic Clusters!";
    validBarrelBCHandle=false;
  }
  
  
  // Get the basic cluster collection in the Endcap 
  bool validEndcapBCHandle=true;
  edm::Handle<edm::View<reco::CaloCluster> > bcEndcapHandle;
  theEvent.getByToken(bcEndcapCollection_, bcEndcapHandle);
  if (!bcEndcapHandle.isValid()) {
    edm::LogError("CoonversionTrackCandidateProducer") 
      << "Error! Can't get the Endcap Basic Clusters";
    validEndcapBCHandle=false; 
  }
  
  

  // Get the Super Cluster collection in the Barrel
  bool validBarrelSCHandle=true;
  edm::Handle<edm::View<reco::CaloCluster> > scBarrelHandle;
  theEvent.getByToken(scHybridBarrelProducer_,scBarrelHandle);
  if (!scBarrelHandle.isValid()) {
    edm::LogError("CoonversionTrackCandidateProducer") 
      << "Error! Can't get the barrel superclusters!";
    validBarrelSCHandle=false;
  }


  // Get the Super Cluster collection in the Endcap
  bool validEndcapSCHandle=true;
  edm::Handle<edm::View<reco::CaloCluster> > scEndcapHandle;
  theEvent.getByToken(scIslandEndcapProducer_,scEndcapHandle);
  if (!scEndcapHandle.isValid()) {
    edm::LogError("CoonversionTrackCandidateProducer") 
      << "Error! Can't get the endcap superclusters!";
    validEndcapSCHandle=false;
  }


  // get the geometry from the event setup:
  theEventSetup.get<CaloGeometryRecord>().get(theCaloGeom_);

  // get Hcal towers collection 
  Handle<CaloTowerCollection> hcalTowersHandle;
  theEvent.getByToken(hcalTowers_, hcalTowersHandle);

  edm::Handle<EcalRecHitCollection> ecalhitsCollEB;
  edm::Handle<EcalRecHitCollection> ecalhitsCollEE;

  theEvent.getByToken(endcapecalCollection_, ecalhitsCollEE);
  theEvent.getByToken(barrelecalCollection_, ecalhitsCollEB);

  edm::ESHandle<EcalSeverityLevelAlgo> sevlv;
  theEventSetup.get<EcalSeverityLevelAlgoRcd>().get(sevlv);
  const EcalSeverityLevelAlgo* sevLevel = sevlv.product();

  caloPtrVecOutIn_.clear();
  caloPtrVecInOut_.clear();

  bool isBarrel=true;
  if ( validBarrelBCHandle && validBarrelSCHandle ) 
    buildCollections(isBarrel, scBarrelHandle, bcBarrelHandle, ecalhitsCollEB, *ecalhitsCollEB, sevLevel, hcalTowersHandle, *outInTrackCandidate_p, *inOutTrackCandidate_p, caloPtrVecOutIn_, caloPtrVecInOut_);

  if ( validEndcapBCHandle && validEndcapSCHandle ) {
    isBarrel=false; 
    buildCollections(isBarrel, scEndcapHandle, bcEndcapHandle, ecalhitsCollEE, *ecalhitsCollEE, sevLevel, hcalTowersHandle, *outInTrackCandidate_p, *inOutTrackCandidate_p, caloPtrVecOutIn_, caloPtrVecInOut_);
  }


  //  std::cout  << "  ConversionTrackCandidateProducer  caloPtrVecOutIn_ size " <<  caloPtrVecOutIn_.size() << " caloPtrVecInOut_ size " << caloPtrVecInOut_.size()  << "\n"; 
  


  // put all products in the event
 // Barrel
 //std::cout  << "ConversionTrackCandidateProducer Putting in the event " << (*outInTrackCandidate_p).size() << " Out In track Candidates " << "\n";
 const edm::OrphanHandle<TrackCandidateCollection> refprodOutInTrackC = theEvent.put( outInTrackCandidate_p, OutInTrackCandidateCollection_ );
 //std::cout  << "ConversionTrackCandidateProducer  refprodOutInTrackC size  " <<  (*(refprodOutInTrackC.product())).size()  <<  "\n";
 //
 //std::cout  << "ConversionTrackCandidateProducer Putting in the event  " << (*inOutTrackCandidate_p).size() << " In Out track Candidates " <<  "\n";
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

 theOutInSeedFinder_->clear();
 theInOutSeedFinder_->clear();
 

  
}


void ConversionTrackCandidateProducer::buildCollections(bool isBarrel, 
							const edm::Handle<edm::View<reco::CaloCluster> > & scHandle,
							const edm::Handle<edm::View<reco::CaloCluster> > & bcHandle,
							edm::Handle<EcalRecHitCollection> ecalRecHitHandle, 
							const EcalRecHitCollection& ecalRecHits,
							const EcalSeverityLevelAlgo* sevLevel,
							//edm::ESHandle<EcalChannelStatus>  chStatus,
							//const EcalChannelStatus* chStatus,
							const edm::Handle<CaloTowerCollection> & hcalTowersHandle,
							TrackCandidateCollection& outInTrackCandidates,
							TrackCandidateCollection& inOutTrackCandidates,
							std::vector<edm::Ptr<reco::CaloCluster> >& vecRecOI,
							std::vector<edm::Ptr<reco::CaloCluster> >& vecRecIO )

{

  //  std::cout << "ConversionTrackCandidateProducer builcollections bc size " << bcHandle->size() <<  "\n";
  //const CaloGeometry* geometry = theCaloGeom_.product();

  //  Loop over SC in the barrel and reconstruct converted photons
  for (unsigned i = 0; i < scHandle->size(); ++i ) {

    reco::CaloClusterPtr aClus= scHandle->ptrAt(i);
  
    // preselection based in Et and H/E cut. 
    if (aClus->energy()/cosh(aClus->eta()) <= minSCEt_) continue;
    if (aClus->eta() > 1.479 && aClus->eta() < 1.556 ) continue;

    const reco::CaloCluster*   pClus=&(*aClus);
    const reco::SuperCluster*  sc=dynamic_cast<const reco::SuperCluster*>(pClus);
    double scEt = sc->energy()/cosh(sc->eta());  
    const CaloTowerCollection* hcalTowersColl = hcalTowersHandle.product();
    EgammaTowerIsolation towerIso(hOverEConeSize_,0.,0.,-1,hcalTowersColl) ;
    double HoE = towerIso.getTowerESum(sc)/sc->energy();
    if (HoE >= maxHOverE_)  continue;

    //// Apply also ecal isolation
    EgammaRecHitIsolation ecalIso(isoConeR_,     
				  isoInnerConeR_, 
				  isoEtaSlice_,  
				  isoEtMin_,    
				  isoEMin_,    
				  theCaloGeom_,
				  ecalRecHits,
				  sevLevel,
				  DetId::Ecal);

    ecalIso.setVetoClustered(vetoClusteredHits_);
    ecalIso.setUseNumCrystals(useNumXtals_);
    if (isBarrel) {
      ecalIso.doFlagChecks(flagsexclEB_);
      ecalIso.doSeverityChecks(ecalRecHitHandle.product(), severitiesexclEB_);
    } else {
      ecalIso.doFlagChecks(flagsexclEE_);
      ecalIso.doSeverityChecks(ecalRecHitHandle.product(), severitiesexclEE_);
    }

    double ecalIsolation = ecalIso.getEtSum(sc);
    if ( ecalIsolation >   ecalIsoCut_offset_ + ecalIsoCut_slope_*scEt ) continue;

    // Now launch the seed finding
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


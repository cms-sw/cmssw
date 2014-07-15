#include "RecoParticleFlow/PFClusterProducer/plugins/PFCTRecHitProducer.h"

#include <memory>

#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CondFormats/DataRecord/interface/HcalChannelQualityRcd.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"


#include "Geometry/CaloTopology/interface/CaloTowerTopology.h"
#include "RecoCaloTools/Navigation/interface/CaloTowerNavigator.h"

#include "RecoLocalCalo/HcalRecAlgos/interface/HcalCaloFlagLabels.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputer.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputerRcd.h"


using namespace std;
using namespace edm;


PFCTRecHitProducer::PFCTRecHitProducer(const edm::ParameterSet& iConfig)
{
  thresh_Barrel_ = 
    iConfig.getParameter<double>("thresh_Barrel");
  thresh_Endcap_ = 
    iConfig.getParameter<double>("thresh_Endcap");

   
  thresh_HF_ = 
    iConfig.getParameter<double>("thresh_HF");
  navigation_HF_ = 
    iConfig.getParameter<bool>("navigation_HF");
  weight_HFem_ =
    iConfig.getParameter<double>("weight_HFem");
  weight_HFhad_ =
    iConfig.getParameter<double>("weight_HFhad");

  HCAL_Calib_ =
    iConfig.getParameter<bool>("HCAL_Calib");
  HF_Calib_ =
    iConfig.getParameter<bool>("HF_Calib");
  HCAL_Calib_29 = 
    iConfig.getParameter<double>("HCAL_Calib_29");
  HF_Calib_29 = 
    iConfig.getParameter<double>("HF_Calib_29");

  shortFibre_Cut = iConfig.getParameter<double>("ShortFibre_Cut");
  longFibre_Fraction = iConfig.getParameter<double>("LongFibre_Fraction");

  longFibre_Cut = iConfig.getParameter<double>("LongFibre_Cut");
  shortFibre_Fraction = iConfig.getParameter<double>("ShortFibre_Fraction");

  applyLongShortDPG_ = iConfig.getParameter<bool>("ApplyLongShortDPG");

  longShortFibre_Cut = iConfig.getParameter<double>("LongShortFibre_Cut");
  minShortTiming_Cut = iConfig.getParameter<double>("MinShortTiming_Cut");
  maxShortTiming_Cut = iConfig.getParameter<double>("MaxShortTiming_Cut");
  minLongTiming_Cut = iConfig.getParameter<double>("MinLongTiming_Cut");
  maxLongTiming_Cut = iConfig.getParameter<double>("MaxLongTiming_Cut");

  applyTimeDPG_ = iConfig.getParameter<bool>("ApplyTimeDPG");
  applyPulseDPG_ = iConfig.getParameter<bool>("ApplyPulseDPG");
  HcalMaxAllowedHFLongShortSev_ = iConfig.getParameter<int>("HcalMaxAllowedHFLongShortSev");
  HcalMaxAllowedHFDigiTimeSev_ = iConfig.getParameter<int>("HcalMaxAllowedHFDigiTimeSev");
  HcalMaxAllowedHFInTimeWindowSev_ = iConfig.getParameter<int>("HcalMaxAllowedHFInTimeWindowSev");
  HcalMaxAllowedChannelStatusSev_ = iConfig.getParameter<int>("HcalMaxAllowedChannelStatusSev");

  ECAL_Compensate_ = iConfig.getParameter<bool>("ECAL_Compensate");
  ECAL_Threshold_ = iConfig.getParameter<double>("ECAL_Threshold");
  ECAL_Compensation_ = iConfig.getParameter<double>("ECAL_Compensation");
  ECAL_Dead_Code_ = iConfig.getParameter<unsigned int>("ECAL_Dead_Code");

  EM_Depth_ = iConfig.getParameter<double>("EM_Depth");
  HAD_Depth_ = iConfig.getParameter<double>("HAD_Depth");

  //Get integer values of individual HCAL HF flags
  hcalHFLongShortFlagValue_=1<<HcalCaloFlagLabels::HFLongShort;
  hcalHFDigiTimeFlagValue_=1<<HcalCaloFlagLabels::HFDigiTime;
  hcalHFInTimeWindowFlagValue_=1<<HcalCaloFlagLabels::HFInTimeWindow;
 
  hcalToken_ = consumes<HBHERecHitCollection>(iConfig.getParameter<InputTag>("hcalRecHitsHBHE"));
  hfToken_ = consumes<HFRecHitCollection>(iConfig.getParameter<InputTag>("hcalRecHitsHF"));
  towersToken_ = consumes<CaloTowerCollection>(iConfig.getParameter<InputTag>("caloTowers"));

					       
  edm::ParameterSet navSet = iConfig.getParameter<edm::ParameterSet>("navigator");
  navigator_ = PFRecHitNavigationFactory::get()->create(navSet.getParameter<std::string>("name"),navSet);
					  
  //--ab
  produces<reco::PFRecHitCollection>();
  produces<reco::PFRecHitCollection>("Cleaned");
  produces<reco::PFRecHitCollection>("HFHAD").setBranchAlias("HFHADRecHits");
  produces<reco::PFRecHitCollection>("HFEM").setBranchAlias("HFEMRecHits");
  //--ab
  
}


void PFCTRecHitProducer::produce(edm::Event& iEvent, 
			       const edm::EventSetup& iSetup) {

  navigator_->beginEvent(iSetup);

  edm::ESHandle<CaloGeometry> geoHandle;
  iSetup.get<CaloGeometryRecord>().get(geoHandle);
  
  // get the hcalBarrel geometry
  const CaloSubdetectorGeometry *hcalBarrelGeometry = 
    geoHandle->getSubdetectorGeometry(DetId::Hcal, HcalBarrel);

  // get the endcap geometry
  const CaloSubdetectorGeometry *hcalEndcapGeometry = 
    geoHandle->getSubdetectorGeometry(DetId::Hcal, HcalEndcap);

  // Get Hcal Severity Level Computer, so that the severity of each rechit flag/status may be determined
  edm::ESHandle<HcalSeverityLevelComputer> hcalSevLvlComputerHndl;
  iSetup.get<HcalSeverityLevelComputerRcd>().get(hcalSevLvlComputerHndl);
  const HcalSeverityLevelComputer* hcalSevLvlComputer = hcalSevLvlComputerHndl.product();


  auto_ptr< vector<reco::PFRecHit> > rechits( new vector<reco::PFRecHit> ); 
  auto_ptr< vector<reco::PFRecHit> > rechitsCleaned( new vector<reco::PFRecHit> ); 
  auto_ptr< vector<reco::PFRecHit> > HFHADRecHits( new vector<reco::PFRecHit> ); 
  auto_ptr< vector<reco::PFRecHit> > HFEMRecHits( new vector<reco::PFRecHit> ); 

  edm::Handle<CaloTowerCollection> caloTowers; 
  iEvent.getByToken(towersToken_,caloTowers);
  edm::Handle<HFRecHitCollection>  hfHandle;  
  iEvent.getByToken(hfToken_,hfHandle);

  edm::Handle<HBHERecHitCollection>  hbheHandle;  
  iEvent.getByToken(hcalToken_,hbheHandle);
  // create rechits
  typedef CaloTowerCollection::const_iterator ICT;
    
  for(ICT ict=caloTowers->begin(); ict!=caloTowers->end();ict++) {
    const CaloTower& ct = (*ict);
	  
	  
    // get the hadronic energy.
    
    // Mike: Just ask for the Hadronic part only now!
    // Patrick : ARGH ! While this is ok for the HCAL, this is 
    // just wrong for the HF (in which em/had are artificially 
    // separated. 
    double energy = ct.hadEnergy();
    //Auguste: Photons in HF have no hadEnergy in fastsim: -> all RecHit collections are empty with photons.
    double energyEM = ct.emEnergy(); // For HF !
    //so test the total energy to deal with the photons in  HF:
    if( (energy+energyEM) < 1e-9 ) continue;
	  
    //get the constituents of the tower
    const std::vector<DetId>& hits = ct.constituents();
    const std::vector<DetId>& allConstituents = theTowerConstituentsMap->constituentsOf(ct.id());
    HcalDetId detid;
    bool foundHCALConstituent = false;
    //Loop on the calotower constituents and search for HCAL
    double dead = 0.;
    double alive = 0.;
    for(unsigned int i=0;i< hits.size();++i) {
      if(hits[i].det()==DetId::Hcal) { 
	foundHCALConstituent = true;
	detid = hits[i];
	// An HCAL tower was found: Look for dead ECAL channels in the same CaloTower.
	if ( ECAL_Compensate_ && energy > ECAL_Threshold_ ) {
	  for(unsigned int j=0;j<allConstituents.size();++j) { 
	    if ( allConstituents[j].det()==DetId::Ecal ) { 
	      alive += 1.;
	      EcalChannelStatus::const_iterator chIt = theEcalChStatus->find(allConstituents[j]);
	      unsigned int dbStatus = chIt != theEcalChStatus->end() ? chIt->getStatusCode() : 0;
	      if ( dbStatus > ECAL_Dead_Code_ ) dead += 1.;
	    }
	  }
	} 
	// Protection: tower 29 in HF is merged with tower 30. 
	// Just take the position of tower 30 in that case. 
	if ( detid.subdet() == HcalForward && abs(detid.ieta()) == 29 ) continue; 
	break;
      }
    }

    // In case of dead ECAL channel, rescale the HCAL energy...
    double rescaleFactor = alive > 0. ? 1. + ECAL_Compensation_*dead/alive : 1.;
	  
    reco::PFRecHit* pfrh = 0;
    reco::PFRecHit* pfrhCleaned = 0;
    //---ab: need 2 rechits for the HF:
    reco::PFRecHit* pfrhHFEM = 0;
    reco::PFRecHit* pfrhHFHAD = 0;
    reco::PFRecHit* pfrhHFEMCleaned = 0;
    reco::PFRecHit* pfrhHFHADCleaned = 0;
    reco::PFRecHit* pfrhHFEMCleaned29 = 0;
    reco::PFRecHit* pfrhHFHADCleaned29 = 0;
  
    if(foundHCALConstituent)
      {
	// std::cout << ", new Energy = " << energy << std::endl;
	switch( detid.subdet() ) {
	case HcalBarrel: 
	  {
	    if(energy < thresh_Barrel_ ) continue;
		if ( rescaleFactor > 1. ) { 
		  pfrhCleaned = createHcalRecHit( detid, 
						  energy, 
						  PFLayer::HCAL_BARREL1, 
						  hcalBarrelGeometry,
						  ct.id() );
		  pfrhCleaned->setTime(rescaleFactor);
		  energy *= rescaleFactor;
		}
		pfrh = createHcalRecHit( detid, 
					 energy, 
					 PFLayer::HCAL_BARREL1, 
					 hcalBarrelGeometry,
					 ct.id() );
		pfrh->setTime(rescaleFactor);
	      }
	      break;
	    case HcalEndcap:
	      {
		if(energy < thresh_Endcap_ ) continue;
		// Apply tower 29 calibration
		if ( HCAL_Calib_ && abs(detid.ieta()) == 29 ) energy *= HCAL_Calib_29;
		if ( rescaleFactor > 1. ) { 
		  pfrhCleaned = createHcalRecHit( detid, 
						  energy, 
						  PFLayer::HCAL_ENDCAP, 
						  hcalEndcapGeometry,
						  ct.id() );
		  pfrhCleaned->setTime(rescaleFactor);
		  energy *= rescaleFactor;
		}
		pfrh = createHcalRecHit( detid, 
					 energy, 
					 PFLayer::HCAL_ENDCAP, 
					 hcalEndcapGeometry,
					 ct.id() );
		pfrh->setTime(rescaleFactor);
	      }
	      break;
	    case HcalOuter:
	      {
	      }
	      break;
	    case HcalForward:
	      {
		//---ab: 2 rechits for HF:
		//double energyemHF = weight_HFem_*ct.emEnergy();
		//double energyhadHF = weight_HFhad_*ct.hadEnergy();
		double energyemHF = weight_HFem_ * energyEM;
		double energyhadHF = weight_HFhad_ * energy;
		// Some energy in the tower !
		if((energyemHF+energyhadHF) < thresh_HF_ ) continue;

		// Some cleaning in the HF 
		double longFibre = energyemHF + energyhadHF/2.;
		double shortFibre = energyhadHF/2.;
		int ieta = detid.ieta();
		int iphi = detid.iphi();
		HcalDetId theLongDetId (HcalForward, ieta, iphi, 1);
		HcalDetId theShortDetId (HcalForward, ieta, iphi, 2);
		typedef HFRecHitCollection::const_iterator iHF;
		iHF theLongHit = hfHandle->find(theLongDetId); 
		iHF theShortHit = hfHandle->find(theShortDetId); 
		// 
		double theLongHitEnergy = 0.;
		double theShortHitEnergy = 0.;
		bool flagShortDPG =  false; 
		bool flagLongDPG = false; 
		bool flagShortTimeDPG = false; 
		bool flagLongTimeDPG = false;
		bool flagShortPulseDPG = false;
		bool flagLongPulseDPG = false;
		//
		if ( theLongHit != hfHandle->end() ) { 
		  int theLongFlag = theLongHit->flags();
		  theLongHitEnergy = theLongHit->energy();
		  flagLongDPG = applyLongShortDPG_ && ( hcalSevLvlComputer->getSeverityLevel(theLongDetId, theLongFlag & hcalHFLongShortFlagValue_, 0)> HcalMaxAllowedHFLongShortSev_);
		  flagLongTimeDPG = applyTimeDPG_ && ( hcalSevLvlComputer->getSeverityLevel(theLongDetId, theLongFlag & hcalHFInTimeWindowFlagValue_, 0)> HcalMaxAllowedHFInTimeWindowSev_);
		  flagLongPulseDPG = applyPulseDPG_ && ( hcalSevLvlComputer->getSeverityLevel(theLongDetId, theLongFlag & hcalHFDigiTimeFlagValue_, 0)> HcalMaxAllowedHFDigiTimeSev_);

		  //flagLongDPG =  applyLongShortDPG_ && theLongHit->flagField(HcalCaloFlagLabels::HFLongShort)==1;
		  //flagLongTimeDPG = applyTimeDPG_ && theLongHit->flagField(HcalCaloFlagLabels::HFInTimeWindow)==1;
		  //flagLongPulseDPG = applyPulseDPG_ && theLongHit->flagField(HcalCaloFlagLabels::HFDigiTime)==1;
		}
		//
		if ( theShortHit != hfHandle->end() ) { 
		  int theShortFlag = theShortHit->flags();
		  theShortHitEnergy = theShortHit->energy();
		  flagShortDPG = applyLongShortDPG_ && ( hcalSevLvlComputer->getSeverityLevel(theShortDetId, theShortFlag & hcalHFLongShortFlagValue_, 0)> HcalMaxAllowedHFLongShortSev_);
		  flagShortTimeDPG = applyTimeDPG_ && ( hcalSevLvlComputer->getSeverityLevel(theShortDetId, theShortFlag & hcalHFInTimeWindowFlagValue_, 0)> HcalMaxAllowedHFInTimeWindowSev_);
		  flagShortPulseDPG = applyPulseDPG_ && ( hcalSevLvlComputer->getSeverityLevel(theShortDetId, theShortFlag & hcalHFDigiTimeFlagValue_, 0)> HcalMaxAllowedHFDigiTimeSev_);
		  //flagShortDPG =  applyLongShortDPG_ && theShortHit->flagField(HcalCaloFlagLabels::HFLongShort)==1;
		  //flagShortTimeDPG = applyTimeDPG_ && theShortHit->flagField(HcalCaloFlagLabels::HFInTimeWindow)==1;
		  //flagShortPulseDPG = applyPulseDPG_ && theShortHit->flagField(HcalCaloFlagLabels::HFDigiTime)==1;
		}

		// Then check the timing in short and long fibres in all other towers.
		if ( theShortHitEnergy > longShortFibre_Cut && 
		     ( theShortHit->time() < minShortTiming_Cut ||
		       theShortHit->time() > maxShortTiming_Cut || 
		       flagShortTimeDPG || flagShortPulseDPG ) ) { 
		  // rescaleFactor = 0. ;
		  pfrhHFHADCleaned = createHcalRecHit( detid, 
						       theShortHitEnergy, 
						       PFLayer::HF_HAD, 
						       hcalEndcapGeometry,
						       ct.id() );
		  pfrhHFHADCleaned->setTime(theShortHit->time());
		  /*
		  std::cout << "ieta/iphi = " << ieta << " " << iphi 
			    << ", Energy em/had/long/short = " 
			    << energyemHF << " " << energyhadHF << " "
			    << theLongHitEnergy << " " << theShortHitEnergy << " " 
			    << ". Time = " << theShortHit->time() << " " 
			    << ". The short and long flags : " 
			    << flagShortDPG << " " << flagLongDPG << " "  
			    << flagShortTimeDPG << " " << flagLongTimeDPG << " "  
			    << flagShortPulseDPG << " " << flagLongPulseDPG << " "  
			    << ". Short fibres were cleaned." << std::endl;
		  */
		  shortFibre -= theShortHitEnergy;
		  theShortHitEnergy = 0.;
		}
		
		
		if ( theLongHitEnergy > longShortFibre_Cut && 
		     ( theLongHit->time() < minLongTiming_Cut ||
		       theLongHit->time() > maxLongTiming_Cut  || 
		       flagLongTimeDPG || flagLongPulseDPG ) ) { 
		  //rescaleFactor = 0. ;
		  pfrhHFEMCleaned = createHcalRecHit( detid, 
						      theLongHitEnergy, 
						      PFLayer::HF_EM, 
						      hcalEndcapGeometry,
						      ct.id());
		  pfrhHFEMCleaned->setTime(theLongHit->time());
		  /*
		  std::cout << "ieta/iphi = " << ieta << " " << iphi 
			    << ", Energy em/had/long/short = " 
			    << energyemHF << " " << energyhadHF << " "
			    << theLongHitEnergy << " " << theShortHitEnergy << " " 
			    << ". Time = " << theLongHit->time() << " " 
			    << ". The short and long flags : " 
			    << flagShortDPG << " " << flagLongDPG << " "  
			    << flagShortTimeDPG << " " << flagLongTimeDPG << " "  
			    << flagShortPulseDPG << " " << flagLongPulseDPG << " "  
			    << ". Long fibres were cleaned." << std::endl;
		  */
		  longFibre -= theLongHitEnergy;
		  theLongHitEnergy = 0.;
		}

		// Some energy must be in the long fibres is there is some energy in the short fibres ! 
		if ( theShortHitEnergy > shortFibre_Cut && 
		     ( theLongHitEnergy/theShortHitEnergy < longFibre_Fraction || 
		       flagShortDPG ) ) {
		  // Check if the long-fibre hit was not cleaned already (because hot)
		  // In this case don't apply the cleaning
		  const HcalChannelStatus* theStatus = theHcalChStatus->getValues(theLongDetId);
		  unsigned theStatusValue = theStatus->getValue();
		  int theSeverityLevel = hcalSevLvlComputer->getSeverityLevel(detid, 0, theStatusValue);
		  // The channel is killed
		  /// if ( !theStatusValue ) 
		  if (theSeverityLevel<=HcalMaxAllowedChannelStatusSev_) {
		    // rescaleFactor = 0. ;
		    pfrhHFHADCleaned = createHcalRecHit( detid, 
							 theShortHitEnergy, 
							 PFLayer::HF_HAD, 
							 hcalEndcapGeometry,
							 ct.id() );
		    pfrhHFHADCleaned->setTime(theShortHit->time());
		    /*
		    std::cout << "ieta/iphi = " << ieta << " " << iphi 
			      << ", Energy em/had/long/short = " 
			      << energyemHF << " " << energyhadHF << " "
			      << theLongHitEnergy << " " << theShortHitEnergy << " " 
			      << ". Time = " << theShortHit->time() << " " 
			      << ". The status value is " << theStatusValue
			      << ". The short and long flags : " 
			      << flagShortDPG << " " << flagLongDPG << " "  
			      << flagShortTimeDPG << " " << flagLongTimeDPG << " "  
			      << flagShortPulseDPG << " " << flagLongPulseDPG << " "  
			      << ". Short fibres were cleaned." << std::endl;
		    */
		    shortFibre -= theShortHitEnergy;
		    theShortHitEnergy = 0.;
		  }
		}

		if ( theLongHitEnergy > longFibre_Cut && 
		     ( theShortHitEnergy/theLongHitEnergy < shortFibre_Fraction || 
		       flagLongDPG ) ) {
		  // Check if the long-fibre hit was not cleaned already (because hot)
		  // In this case don't apply the cleaning
		  const HcalChannelStatus* theStatus = theHcalChStatus->getValues(theShortDetId);
		  unsigned theStatusValue = theStatus->getValue();

		  int theSeverityLevel = hcalSevLvlComputer->getSeverityLevel(detid, 0, theStatusValue);
		  // The channel is killed
		  /// if ( !theStatusValue ) 
		  if (theSeverityLevel<=HcalMaxAllowedChannelStatusSev_) {
		    
		    //rescaleFactor = 0. ;
		    pfrhHFEMCleaned = createHcalRecHit( detid, 
						      theLongHitEnergy, 
						      PFLayer::HF_EM, 
						      hcalEndcapGeometry,
						      ct.id() );
		    pfrhHFEMCleaned->setTime(theLongHit->time());
		    /*
		    std::cout << "ieta/iphi = " << ieta << " " << iphi 
			      << ", Energy em/had/long/short = " 
			      << energyemHF << " " << energyhadHF << " "
			      << theLongHitEnergy << " " << theShortHitEnergy << " " 
			      << ". The status value is " << theStatusValue
			      << ". Time = " << theLongHit->time() << " " 
			      << ". The short and long flags : " 
			      << flagShortDPG << " " << flagLongDPG << " "  
			      << flagShortTimeDPG << " " << flagLongTimeDPG << " "  
			      << flagShortPulseDPG << " " << flagLongPulseDPG << " "  
			      << ". Long fibres were cleaned." << std::endl;
		    */
		    longFibre -= theLongHitEnergy;
		    theLongHitEnergy = 0.;
		  }
		}

		// Special treatment for tower 29
		// A tower with energy only at ieta = +/- 29 is not physical -> Clean
		if ( abs(ieta) == 29 ) { 
		  // rescaleFactor = 0. ;
		  // Clean long fibres
		  if ( theLongHitEnergy > shortFibre_Cut/2. ) { 
		    pfrhHFEMCleaned29 = createHcalRecHit( detid, 
							  theLongHitEnergy, 
							  PFLayer::HF_EM, 
							  hcalEndcapGeometry,
							  ct.id() );
		    pfrhHFEMCleaned29->setTime(theLongHit->time());
		    /*
		    std::cout << "ieta/iphi = " << ieta << " " << iphi 
			      << ", Energy em/had/long/short = " 
			      << energyemHF << " " << energyhadHF << " "
			      << theLongHitEnergy << " " << theShortHitEnergy << " " 
			      << ". Long fibres were cleaned." << std::endl;
		    */
		    longFibre -= theLongHitEnergy;
		    theLongHitEnergy = 0.;
		  }
		  // Clean short fibres
		  if ( theShortHitEnergy > shortFibre_Cut/2. ) { 
		    pfrhHFHADCleaned29 = createHcalRecHit( detid, 
							   theShortHitEnergy, 
							   PFLayer::HF_HAD, 
							   hcalEndcapGeometry,
							   ct.id());
		    pfrhHFHADCleaned29->setTime(theShortHit->time());
		    /*
		    std::cout << "ieta/iphi = " << ieta << " " << iphi 
			      << ", Energy em/had/long/short = " 
			      << energyemHF << " " << energyhadHF << " "
			      << theLongHitEnergy << " " << theShortHitEnergy << " " 
			      << ". Short fibres were cleaned." << std::endl;
		    */
		    shortFibre -= theShortHitEnergy;
		    theShortHitEnergy = 0.;
		  }
		}
		// Check the timing of the long and short fibre rechits
		
		// First, check the timing of long and short fibre in eta = 29 if tower 30.
		else if ( abs(ieta) == 30 ) { 
		  int ieta29 = ieta > 0 ? 29 : -29;
		  HcalDetId theLongDetId29 (HcalForward, ieta29, iphi, 1);
		  HcalDetId theShortDetId29 (HcalForward, ieta29, iphi, 2);
		  iHF theLongHit29 = hfHandle->find(theLongDetId29); 
		  iHF theShortHit29 = hfHandle->find(theShortDetId29); 
		  // 
		  double theLongHitEnergy29 = 0.;
		  double theShortHitEnergy29 = 0.;
		  bool flagShortDPG29 =  false; 
		  bool flagLongDPG29 = false; 
		  bool flagShortTimeDPG29 = false; 
		  bool flagLongTimeDPG29 = false;
		  bool flagShortPulseDPG29 = false;
		  bool flagLongPulseDPG29 = false;
		  //
		  if ( theLongHit29 != hfHandle->end() ) { 
		    int theLongFlag29 = theLongHit29->flags();
		    theLongHitEnergy29 = theLongHit29->energy() ;
		    flagLongDPG29 = applyLongShortDPG_ && ( hcalSevLvlComputer->getSeverityLevel(theLongDetId29, theLongFlag29 & hcalHFLongShortFlagValue_, 0)> HcalMaxAllowedHFLongShortSev_);
		    flagLongTimeDPG29 = applyTimeDPG_ && ( hcalSevLvlComputer->getSeverityLevel(theLongDetId29, theLongFlag29 & hcalHFInTimeWindowFlagValue_, 0)> HcalMaxAllowedHFInTimeWindowSev_);
		    flagLongPulseDPG29 = applyPulseDPG_ && ( hcalSevLvlComputer->getSeverityLevel(theLongDetId29, theLongFlag29 & hcalHFDigiTimeFlagValue_, 0)> HcalMaxAllowedHFDigiTimeSev_);

		    //flagLongDPG29 = applyLongShortDPG_ && theLongHit29->flagField(HcalCaloFlagLabels::HFLongShort)==1;
		    //flagLongTimeDPG29 = applyTimeDPG_ && theLongHit29->flagField(HcalCaloFlagLabels::HFInTimeWindow)==1;
		    //flagLongPulseDPG29 = applyPulseDPG_ && theLongHit29->flagField(HcalCaloFlagLabels::HFDigiTime)==1;
		  }
		  //
		  if ( theShortHit29 != hfHandle->end() ) { 
		    int theShortFlag29 = theShortHit29->flags();
		    theShortHitEnergy29 = theShortHit29->energy();		  
		    flagShortDPG29 = applyLongShortDPG_ && ( hcalSevLvlComputer->getSeverityLevel(theShortDetId29, theShortFlag29 & hcalHFLongShortFlagValue_, 0)> HcalMaxAllowedHFLongShortSev_);
		    flagShortTimeDPG29 = applyTimeDPG_ && ( hcalSevLvlComputer->getSeverityLevel(theShortDetId29, theShortFlag29 & hcalHFInTimeWindowFlagValue_, 0)> HcalMaxAllowedHFInTimeWindowSev_);
		    flagLongPulseDPG29 = applyPulseDPG_ && ( hcalSevLvlComputer->getSeverityLevel(theShortDetId29, theShortFlag29 & hcalHFDigiTimeFlagValue_, 0)> HcalMaxAllowedHFDigiTimeSev_);

		    //flagShortDPG29 = applyLongShortDPG_ && theShortHit29->flagField(HcalCaloFlagLabels::HFLongShort)==1;
		    //flagShortTimeDPG29 = applyTimeDPG_ && theShortHit29->flagField(HcalCaloFlagLabels::HFInTimeWindow)==1;
		    //flagShortPulseDPG29 = applyPulseDPG_ && theShortHit29->flagField(HcalCaloFlagLabels::HFDigiTime)==1;
		  }

		  if ( theLongHitEnergy29 > longShortFibre_Cut && 
		       ( theLongHit29->time() < minLongTiming_Cut ||
		         theLongHit29->time() > maxLongTiming_Cut ||
			 flagLongTimeDPG29 || flagLongPulseDPG29 ) ) { 
		    //rescaleFactor = 0. ;
		    pfrhHFEMCleaned29 = createHcalRecHit( detid, 
							  theLongHitEnergy29, 
							  PFLayer::HF_EM, 
							  hcalEndcapGeometry,
							  ct.id() );
		    pfrhHFEMCleaned29->setTime(theLongHit29->time());
		    /*
		    std::cout << "ieta/iphi = " << ieta29 << " " << iphi 
			      << ", Energy em/had/long/short = " 
			      << energyemHF << " " << energyhadHF << " "
			      << theLongHitEnergy29 << " " << theShortHitEnergy29 << " " 
			      << ". Time = " << theLongHit29->time() << " " 
			      << ". The short and long flags : " 
			      << flagShortDPG29 << " " << flagLongDPG29 << " "  
			      << flagShortTimeDPG29 << " " << flagLongTimeDPG29 << " "  
			      << flagShortPulseDPG29 << " " << flagLongPulseDPG29 << " "  
			      << ". Long fibres were cleaned." << std::endl;
		    */
		    longFibre -= theLongHitEnergy29;
		    theLongHitEnergy29 = 0;
		  }

		  if ( theShortHitEnergy29 > longShortFibre_Cut && 
		       ( theShortHit29->time() < minShortTiming_Cut ||
		         theShortHit29->time() > maxShortTiming_Cut ||
			 flagShortTimeDPG29 || flagShortPulseDPG29 ) ) { 
		    //rescaleFactor = 0. ;
		    pfrhHFHADCleaned29 = createHcalRecHit( detid, 
							 theShortHitEnergy29, 
							 PFLayer::HF_HAD, 
							 hcalEndcapGeometry,
							 ct.id() );
		    pfrhHFHADCleaned29->setTime(theShortHit29->time());
		    /*
		    std::cout << "ieta/iphi = " << ieta29 << " " << iphi 
			      << ", Energy em/had/long/short = " 
			      << energyemHF << " " << energyhadHF << " "
			      << theLongHitEnergy29 << " " << theShortHitEnergy29 << " " 
			      << ". Time = " << theShortHit29->time() << " " 
			      << ". The short and long flags : " 
			      << flagShortDPG29 << " " << flagLongDPG29 << " "  
			      << flagShortTimeDPG29 << " " << flagLongTimeDPG29 << " "  
			      << flagShortPulseDPG29 << " " << flagLongPulseDPG29 << " "  
			      << ". Short fibres were cleaned." << std::endl;
		    */
		    shortFibre -= theShortHitEnergy29;
		    theShortHitEnergy29 = 0.;
		  }

		  // Some energy must be in the long fibres is there is some energy in the short fibres ! 
		  if ( theShortHitEnergy29 > shortFibre_Cut && 
		       ( theLongHitEnergy29/theShortHitEnergy29 < 2.*longFibre_Fraction || 
			 flagShortDPG29 ) ) {
		    // Check if the long-fibre hit was not cleaned already (because hot)
		    // In this case don't apply the cleaning
		    const HcalChannelStatus* theStatus = theHcalChStatus->getValues(theLongDetId29);
		    unsigned theStatusValue = theStatus->getValue();

		    int theSeverityLevel = hcalSevLvlComputer->getSeverityLevel(detid, 0, theStatusValue);
		    // The channel is killed
		    /// if ( !theStatusValue ) 
		    if (theSeverityLevel<=HcalMaxAllowedChannelStatusSev_) {
		      //rescaleFactor = 0. ;
		      pfrhHFHADCleaned29 = createHcalRecHit( detid, 
							   theShortHitEnergy29, 
							   PFLayer::HF_HAD, 
							   hcalEndcapGeometry,
							   ct.id() );
		      pfrhHFHADCleaned29->setTime(theShortHit29->time());
		      /*
		      std::cout << "ieta/iphi = " << ieta29 << " " << iphi 
				<< ", Energy em/had/long/short = " 
				<< energyemHF << " " << energyhadHF << " "
				<< theLongHitEnergy29 << " " << theShortHitEnergy29 << " " 
				<< ". Time = " << theShortHit29->time() << " " 
				<< ". The status value is " << theStatusValue
				<< ". The short and long flags : " 
				<< flagShortDPG29 << " " << flagLongDPG29 << " "  
				<< flagShortTimeDPG29 << " " << flagLongTimeDPG29 << " "  
				<< flagShortPulseDPG29 << " " << flagLongPulseDPG29 << " "  
				<< ". Short fibres were cleaned." << std::endl;
		      */
		      shortFibre -= theShortHitEnergy29;
		      theShortHitEnergy29 = 0.;
		    }	    
		  }
		  
		  // Some energy must be in the short fibres is there is some energy in the long fibres ! 
		  if ( theLongHitEnergy29 > longFibre_Cut && 
		       ( theShortHitEnergy29/theLongHitEnergy29 < shortFibre_Fraction || 
			 flagLongDPG29 ) ) {
		    // Check if the long-fibre hit was not cleaned already (because hot)
		    // In this case don't apply the cleaning
		    const HcalChannelStatus* theStatus = theHcalChStatus->getValues(theShortDetId29);
		    unsigned theStatusValue = theStatus->getValue();
		    int theSeverityLevel = hcalSevLvlComputer->getSeverityLevel(detid, 0, theStatusValue);
		    // The channel is killed
		    /// if ( !theStatusValue ) 
		    if (theSeverityLevel<=HcalMaxAllowedChannelStatusSev_) {

		      //rescaleFactor = 0. ;
		      pfrhHFEMCleaned29 = createHcalRecHit( detid, 
							    theLongHitEnergy29, 
							    PFLayer::HF_EM, 
							    hcalEndcapGeometry,
							    ct.id() );
		      pfrhHFEMCleaned29->setTime(theLongHit29->time());
		      /* 
		      std::cout << "ieta/iphi = " << ieta29 << " " << iphi 
				<< ", Energy em/had/long/short = " 
				<< energyemHF << " " << energyhadHF << " "
				<< theLongHitEnergy29 << " " << theShortHitEnergy29 << " " 
				<< ". The status value is " << theStatusValue
				<< ". Time = " << theLongHit29->time() << " " 
				<< ". The short and long flags : " 
				<< flagShortDPG29 << " " << flagLongDPG29 << " "  
				<< flagShortTimeDPG29 << " " << flagLongTimeDPG29 << " "  
				<< flagShortPulseDPG29 << " " << flagLongPulseDPG29 << " "  
				<< ". Long fibres were cleaned." << std::endl;
		      */
		      longFibre -= theLongHitEnergy29;
		      theLongHitEnergy29 = 0.;
		    }
		  }

		  // Check that the energy in tower 29 is smaller than in tower 30
		  // First in long fibres
		  if ( theLongHitEnergy29 > std::max(theLongHitEnergy,shortFibre_Cut/2) ) { 
		    pfrhHFEMCleaned29 = createHcalRecHit( detid, 
							  theLongHitEnergy29, 
							  PFLayer::HF_EM, 
							  hcalEndcapGeometry,
							  ct.id() );
		    pfrhHFEMCleaned29->setTime(theLongHit29->time());
		    /*
		    std::cout << "ieta/iphi = " << ieta29 << " " << iphi 
			      << ", Energy L29/S29/L30/S30 = " 
			      << theLongHitEnergy29 << " " << theShortHitEnergy29 << " "
			      << theLongHitEnergy << " " << theShortHitEnergy << " " 
			      << ". Long fibres were cleaned." << std::endl;
		    */
		    longFibre -= theLongHitEnergy29;
		    theLongHitEnergy29 = 0.;
		  }
		  // Second in short fibres
		  if ( theShortHitEnergy29 > std::max(theShortHitEnergy,shortFibre_Cut/2.) ) { 
		    pfrhHFHADCleaned29 = createHcalRecHit( detid, 
							   theShortHitEnergy29, 
							   PFLayer::HF_HAD, 
							   hcalEndcapGeometry,
							   ct.id() );
		    pfrhHFHADCleaned29->setTime(theShortHit29->time());
		    /*
		    std::cout << "ieta/iphi = " << ieta << " " << iphi 
			      << ", Energy L29/S29/L30/S30 = " 
			      << theLongHitEnergy29 << " " << theShortHitEnergy29 << " "
			      << theLongHitEnergy << " " << theShortHitEnergy << " " 
			      << ". Short fibres were cleaned." << std::endl;
		    */
		    shortFibre -= theShortHitEnergy29;
		    theShortHitEnergy29 = 0.;
		  }
		}


		// Determine EM and HAD after cleaning of short and long fibres
		energyhadHF = 2.*shortFibre;
		energyemHF = longFibre - shortFibre;

		// The EM energy might be negative, as it amounts to Long - Short
		// In that case, put the EM "energy" in the HAD energy
		// Just to avoid systematic positive bias due to "Short" high fluctuations
		if ( energyemHF < thresh_HF_ ) { 
		  energyhadHF += energyemHF;
		  energyemHF = 0.;
		}

		// Apply HCAL calibration factors flor towers close to 29, if requested
		if ( HF_Calib_ && abs(detid.ieta()) <= 32 ) { 
		  energyhadHF *= HF_Calib_29;
		  energyemHF *= HF_Calib_29;
		}
				
		// Create an EM and a HAD rechit if above threshold.
		if ( energyemHF > thresh_HF_ || energyhadHF > thresh_HF_ ) { 
		  pfrhHFEM = createHcalRecHit( detid, 
					       energyemHF, 
					       PFLayer::HF_EM, 
					       hcalEndcapGeometry,
					       ct.id() );
		  pfrhHFHAD = createHcalRecHit( detid, 
						energyhadHF, 
						PFLayer::HF_HAD, 
						hcalEndcapGeometry,
						ct.id() );

		}
		
	      }
	      break;
	    default:
	      LogError("PFCTRecHitProducerHCAL")
		<<"CaloTower constituent: unknown layer : "
		<<detid.subdet()<<endl;
	    } 


	    if(pfrh) { 
	      rechits->push_back( *pfrh );
	      delete pfrh;
	    }
	    if(pfrhCleaned) { 
	      rechitsCleaned->push_back( *pfrhCleaned );
	      delete pfrhCleaned;
	    }
	    if(pfrhHFEM) { 
	      HFEMRecHits->push_back( *pfrhHFEM );
	      delete pfrhHFEM;
	    }
	    if(pfrhHFHAD) { 
	      HFHADRecHits->push_back( *pfrhHFHAD );
	      delete pfrhHFHAD;
	    }
	    if(pfrhHFEMCleaned) { 
	      rechitsCleaned->push_back( *pfrhHFEMCleaned );
	      delete pfrhHFEMCleaned;
	    }
	    if(pfrhHFHADCleaned) { 
	      rechitsCleaned->push_back( *pfrhHFHADCleaned );
	      delete pfrhHFHADCleaned;
	    }
	    if(pfrhHFEMCleaned29) { 
	      rechitsCleaned->push_back( *pfrhHFEMCleaned29 );
	      delete pfrhHFEMCleaned29;
	    }
	    if(pfrhHFHADCleaned29) { 
	      rechitsCleaned->push_back( *pfrhHFHADCleaned29 );
	      delete pfrhHFHADCleaned29;
	    }
      }
  }

   //ok now do navigation
   //create a refprod here

   edm::RefProd<reco::PFRecHitCollection> refProd = 
     iEvent.getRefBeforePut<reco::PFRecHitCollection>();


   for( unsigned int i=0;i<rechits->size();++i) {
     navigator_->associateNeighbours(rechits->at(i),rechits,refProd);
   }

    if   (navigation_HF_) {

      edm::RefProd<reco::PFRecHitCollection> refProdEM = 
	iEvent.getRefBeforePut<reco::PFRecHitCollection>("HFEM");


      for( unsigned int i=0;i<HFEMRecHits->size();++i) {
        navigator_->associateNeighbours(HFEMRecHits->at(i),HFEMRecHits,refProdEM);
      }

      edm::RefProd<reco::PFRecHitCollection> refProdHAD = 
	iEvent.getRefBeforePut<reco::PFRecHitCollection>("HFHAD");


      for( unsigned int i=0;i<HFHADRecHits->size();++i) {
        navigator_->associateNeighbours(HFHADRecHits->at(i),HFHADRecHits,refProdHAD);
      }
    }

  iEvent.put( rechits,"" );	
  iEvent.put( rechitsCleaned,"Cleaned" );	
  iEvent.put( HFEMRecHits,"HFEM" );	
  iEvent.put( HFHADRecHits,"HFHAD" );	

}

PFCTRecHitProducer::~PFCTRecHitProducer() {}

// ------------ method called once each job just before starting event loop  ------------
void 
PFCTRecHitProducer::beginLuminosityBlock(const edm::LuminosityBlock& lumi,
					 const EventSetup& es) {

  // Get cleaned channels in the HCAL and HF 
  // HCAL channel status map ****************************************
  edm::ESHandle<HcalChannelQuality> hcalChStatus;    
  es.get<HcalChannelQualityRcd>().get( hcalChStatus );
  theHcalChStatus = hcalChStatus.product();

  // Retrieve the good/bad ECAL channels from the DB
  edm::ESHandle<EcalChannelStatus> ecalChStatus;
  es.get<EcalChannelStatusRcd>().get(ecalChStatus);
  theEcalChStatus = ecalChStatus.product();

  edm::ESHandle<CaloTowerConstituentsMap> cttopo;
  es.get<IdealGeometryRecord>().get(cttopo);
  theTowerConstituentsMap = cttopo.product();
}


reco::PFRecHit* 
PFCTRecHitProducer::createHcalRecHit( const DetId& detid,
					double energy,
					PFLayer::Layer layer,
					const CaloSubdetectorGeometry* geom,
					const CaloTowerDetId& newDetId ) {
  
  const CaloCellGeometry *thisCell = geom->getGeometry(detid);
  if(!thisCell) {
    edm::LogError("PFRecHitProducerHCAL")
      <<"warning detid "<<detid.rawId()<<" not found in layer "
      <<layer<<endl;
    return 0;
  }
  
  const GlobalPoint& position = thisCell->getPosition();
  
  double depth_correction = 0.;
  switch ( layer ) { 
  case PFLayer::HF_EM:
    depth_correction = position.z() > 0. ? EM_Depth_ : -EM_Depth_;
    break;
  case PFLayer::HF_HAD:
    depth_correction = position.z() > 0. ? HAD_Depth_ : -HAD_Depth_;
    break;
  default:
    break;
  }

  reco::PFRecHit *rh = 
    new reco::PFRecHit( newDetId.rawId(),  layer, energy, 
			position.x(), position.y(), position.z()+depth_correction, 
			0,0,0 );
 
  
  
  
  // set the corners
  const CaloCellGeometry::CornersVec& corners = thisCell->getCorners();

  rh->setNECorner( corners[0].x(), corners[0].y(),  corners[0].z()+depth_correction );
  rh->setSECorner( corners[1].x(), corners[1].y(),  corners[1].z()+depth_correction );
  rh->setSWCorner( corners[2].x(), corners[2].y(),  corners[2].z()+depth_correction );
  rh->setNWCorner( corners[3].x(), corners[3].y(),  corners[3].z()+depth_correction );
 
  return rh;
}

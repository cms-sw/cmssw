#include "RecoParticleFlow/PFClusterProducer/plugins/PFHCALDualTimeRecHitProducer.h"

#include <memory>

#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"

#include "DataFormats/HcalRecHit/interface/HFRecHit.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
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
#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"

#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"

#include "Geometry/CaloTopology/interface/CaloTowerTopology.h"
#include "RecoCaloTools/Navigation/interface/CaloTowerNavigator.h"

#include "RecoLocalCalo/HcalRecAlgos/interface/HcalCaloFlagLabels.h"


using namespace std;
using namespace edm;

PFHCALDualTimeRecHitProducer::PFHCALDualTimeRecHitProducer(const edm::ParameterSet& iConfig)
  : PFRecHitProducer( iConfig ) 
{

 

  // access to the collections of rechits 

  
  inputTagHcalRecHitsHBHE_ =
    iConfig.getParameter<InputTag>("hcalRecHitsHBHE");
    
  inputTagHcalRecHitsHF_ =
    iConfig.getParameter<InputTag>("hcalRecHitsHF");
    
 
  inputTagCaloTowers_ = 
    iConfig.getParameter<InputTag>("caloTowers");
   
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

  ECAL_Compensate_ = iConfig.getParameter<bool>("ECAL_Compensate");
  ECAL_Threshold_ = iConfig.getParameter<double>("ECAL_Threshold");
  ECAL_Compensation_ = iConfig.getParameter<double>("ECAL_Compensation");
  ECAL_Dead_Code_ = iConfig.getParameter<unsigned int>("ECAL_Dead_Code");

  EM_Depth_ = iConfig.getParameter<double>("EM_Depth");
  HAD_Depth_ = iConfig.getParameter<double>("HAD_Depth");

  //--ab
  produces<reco::PFRecHitCollection>("HFHAD").setBranchAlias("HFHADRecHits");
  produces<reco::PFRecHitCollection>("HFEM").setBranchAlias("HFEMRecHits");
  //--ab
}



PFHCALDualTimeRecHitProducer::~PFHCALDualTimeRecHitProducer() {}



void PFHCALDualTimeRecHitProducer::createRecHits(vector<reco::PFRecHit>& rechits,
					 vector<reco::PFRecHit>& rechitsCleaned,
					 edm::Event& iEvent, 
					 const edm::EventSetup& iSetup ) {

  
  // this map is necessary to find the rechit neighbours efficiently
  //C but I should think about using Florian's hashed index to do this.
  //C in which case the map might not be necessary anymore
  //C however the hashed index does not seem to be implemented for HCAL
  // 
  // the key of this map is detId. 
  // the value is the index in the rechits vector
  map<unsigned,  unsigned > idSortedRecHits;
  map<unsigned,  unsigned > idSortedRecHitsHFEM;
  map<unsigned,  unsigned > idSortedRecHitsHFHAD;
  typedef map<unsigned, unsigned >::iterator IDH;  


  edm::ESHandle<CaloGeometry> geoHandle;
  iSetup.get<CaloGeometryRecord>().get(geoHandle);
  
  // get the hcalBarrel geometry
  const CaloSubdetectorGeometry *hcalBarrelGeometry = 
    geoHandle->getSubdetectorGeometry(DetId::Hcal, HcalBarrel);

  // get the endcap geometry
  const CaloSubdetectorGeometry *hcalEndcapGeometry = 
    geoHandle->getSubdetectorGeometry(DetId::Hcal, HcalEndcap);


  // get the hcal topology
  edm::ESHandle<HcalTopology> hcalTopology;
  iSetup.get<HcalRecNumberingRecord>().get( hcalTopology );


  //--ab
  auto_ptr< vector<reco::PFRecHit> > HFHADRecHits( new vector<reco::PFRecHit> ); 
  auto_ptr< vector<reco::PFRecHit> > HFEMRecHits( new vector<reco::PFRecHit> ); 
  //--ab

  // 2 possibilities to make HCAL clustering :
  // - from the HCAL rechits
  // - from the CaloTowers. 
  // ultimately, clustering will be done taking CaloTowers as an 
  // input. This possibility is currently under investigation, and 
  // was thus made optional.

  // in the first step, we will fill the map of PFRecHits hcalrechits
  // either from CaloTowers or from HCAL rechits. 

  // in the second step, we will perform clustering on this map.

  if( !(inputTagCaloTowers_ == InputTag()) ) {
      
    edm::Handle<CaloTowerCollection> caloTowers;
	edm::ESHandle<CaloTowerTopology> caloTowerTopology;
	iSetup.get<HcalRecNumberingRecord>().get(caloTowerTopology);
    const CaloSubdetectorGeometry *caloTowerGeometry = 0; 
    // = geometry_->getSubdetectorGeometry(id)

    // get calotowers
    bool found = iEvent.getByLabel(inputTagCaloTowers_,
				   caloTowers);

    if(!found) {
      ostringstream err;
      err<<"could not find rechits "<<inputTagCaloTowers_;
      LogError("PFHCALDualTimeRecHitProducer")<<err.str()<<endl;
    
      throw cms::Exception( "MissingProduct", err.str());
    }
    else {
      assert( caloTowers.isValid() );

      // get HF rechits
      edm::Handle<HFRecHitCollection>  hfHandle;  
      found = iEvent.getByLabel(inputTagHcalRecHitsHF_,
				hfHandle);
      
      if(!found) {
	ostringstream err;
	err<<"could not find HF rechits "<<inputTagHcalRecHitsHF_;
	LogError("PFHCALDualTimeRecHitProducer")<<err.str()<<endl;
	
	throw cms::Exception( "MissingProduct", err.str());
      }
      else {
	assert( hfHandle.isValid() );
      }
      
      // get HBHE rechits
      edm::Handle<HBHERecHitCollection>  hbheHandle;  
  
      found = iEvent.getByLabel(inputTagHcalRecHitsHBHE_,
				hbheHandle);
      
      if(!found) {
	ostringstream err;
	err<<"could not find HBHE rechits "<<inputTagHcalRecHitsHBHE_;
	LogError("PFHCALDualTimeRecHitProducer")<<err.str()<<endl;
	
	throw cms::Exception( "MissingProduct", err.str());
      }
      else {
	assert( hbheHandle.isValid() );
      }
      
      for(unsigned irechit=0; irechit<hbheHandle->size(); irechit++) {
        const HBHERecHit& hit = (*hbheHandle)[irechit];


        double hitenergy = hit.energy();
	double hittime = hit.time();

        reco::PFRecHit* pfrh = 0;
        reco::PFRecHit* pfrhCleaned = 0;


        const HcalDetId& detid = hit.detid();
        switch( detid.subdet() ) {
        case HcalBarrel:
          {
            if(hitenergy < thresh_Barrel_ ) continue;
            if(detid.depth()==1) {
              hittime -= 48.9580/(2.16078+hitenergy);
            } else if(detid.depth()==2) {
              hittime -= 34.2860/(1.23746+hitenergy);
            } else if(detid.depth()==3) {
              hittime -= 38.6872/(1.48051+hitenergy);
            }
// time window for signal=4
            if(    (detid.depth()==1 && hittime>-20 && hittime<5) 
                || (detid.depth()==2 && hittime>-17 && hittime<8) 
                || (detid.depth()==3 && hittime>-15 && hittime<10) 
              ) {
              pfrh = createHcalRecHit( detid,
                                       hitenergy,
                                       PFLayer::HCAL_BARREL1,
                                       hcalBarrelGeometry );
	      pfrh->setRescale(hittime);
            }
          }
          break;
        case HcalEndcap:
          {
            if(hitenergy < thresh_Endcap_ ) continue;
            // Apply tower 29 calibration
            if ( HCAL_Calib_ && abs(detid.ieta()) == 29 ) hitenergy *= HCAL_Calib_29;
            if(detid.depth()==1) {
              hittime -= 60.8050/(3.07285+hitenergy);
            } else if(detid.depth()==2) {
              hittime -= 47.1677/(2.06485+hitenergy);
            } else if(detid.depth()==3) {
              hittime -= 37.1941/(1.53790+hitenergy);
            } else if(detid.depth()==4) {
              hittime -= 42.9898/(1.92969+hitenergy);
            } else if(detid.depth()==5) {
              hittime -= 48.3157/(2.29903+hitenergy);
            }
// time window for signal=4
            if(    (detid.depth()==1 && hittime>-20 && hittime<5) 
                || (detid.depth()==2 && hittime>-19 && hittime<6) 
                || (detid.depth()==3 && hittime>-18 && hittime<7) 
                || (detid.depth()==4 && hittime>-17 && hittime<8) 
                || (detid.depth()==5 && hittime>-15 && hittime<10) 
              ) {
              pfrh = createHcalRecHit( detid,
                                       hitenergy,
                                       PFLayer::HCAL_ENDCAP,
                                       hcalEndcapGeometry );
	      pfrh->setRescale(hittime);
            }
          }
          break;
        default:
          LogError("PFHCALDualTimeRecHitProducer")
            <<"HCAL rechit: unknown layer : "<<detid.subdet()<<endl;
          continue;
        }

        if(pfrh) {
          rechits.push_back( *pfrh );
          delete pfrh;
          idSortedRecHits.insert( make_pair(detid.rawId(),
                                            rechits.size()-1 ) );
        }
        if(pfrhCleaned) { 
          rechitsCleaned.push_back( *pfrhCleaned );
          delete pfrhCleaned;
        }
      }



      // create rechits
      typedef CaloTowerCollection::const_iterator ICT;
    
      for(ICT ict=caloTowers->begin(); ict!=caloTowers->end();ict++) {
	  
	const CaloTower& ct = (*ict);
	  
	//C	
	if(!caloTowerGeometry) 
	  caloTowerGeometry = geoHandle->getSubdetectorGeometry(ct.id());

	  
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
	  
	assert( ct.constituentsSize() );	  
	//Mike: The DetId will be taken by the first Hadronic constituent
	//         of the tower. That is only what we need

	
	//get the constituents of the tower
	const std::vector<DetId>& hits = ct.constituents();
	const std::vector<DetId>& allConstituents = theTowerConstituentsMap->constituentsOf(ct.id());

	/*
	for(unsigned int i=0;i< hits.size();++i) {
	  if(hits[i].det()==DetId::Hcal) {
	    HcalDetId did = hits[i];
	    if ( did.subdet()==HcalEndcap || did.subdet()==HcalForward ) { 
	      //double en = hits[i].energy();
	      int ieta = did.ieta();
	      const CaloCellGeometry *thisCell = hcalEndcapGeometry->getGeometry(did);
	      const GlobalPoint& position = thisCell->getPosition();
	      if ( abs(ieta) > 27 && abs(ieta) < 33 && energy > 10. ) { 
		std::cout << "HE/HF hit " << i << " at eta = " << ieta 
			  << " with CT energy = " << energy 
			  << " at eta, z (hit) = " << position.eta() << " " << position.z()
			  << " at eta, z (cte) = " << ct.emPosition().eta() << " " << ct.emPosition().z()
			  << " at eta, z (cth) = " << ct.hadPosition().eta() << " " << ct.hadPosition().z()
			  << " at eta, z (cto) = " << ct.eta() << " " << ct.vz() 
			  << std::endl;
	      }
	    }
	  }
	}
	*/
	
	//Reserve the DetId we are looking for:

	HcalDetId detid;
	// EcalDetId edetid;
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
//	double rescaleFactor = alive > 0. ? 1. + ECAL_Compensation_*dead/alive : 1.;
	  
//	reco::PFRecHit* pfrh = 0;
//	reco::PFRecHit* pfrhCleaned = 0;
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

		/*
		// Check the timing
		if ( energy > 5. ) { 
		  for(unsigned int i=0;i< hits.size();++i) {
		    if( hits[i].det() != DetId::Hcal ) continue; 
		    HcalDetId theDetId = hits[i]; 
		    typedef HBHERecHitCollection::const_iterator iHBHE;
		    iHBHE theHit = hbheHandle->find(theDetId); 
		    if ( theHit != hbheHandle->end() ) 
		      std::cout << "HCAL hit : " 
				<< theDetId.ieta() << " " << theDetId.iphi() << " " 
				<< theHit->energy() << " " << theHit->time() << std::endl;
		  }
		}
		*/


		// if ( HCAL_Calib_ ) energy   *= std::min(max_Calib_,myPFCorr->getValues(detid)->getValue());
		//if ( rescaleFactor > 1. ) 
		// std::cout << "Barrel HCAL energy rescaled from = " << energy << " to " << energy*rescaleFactor << std::endl;
/*
		if ( rescaleFactor > 1. ) { 
		  pfrhCleaned = createHcalRecHit( detid, 
						  energy, 
						  PFLayer::HCAL_BARREL1, 
						  hcalBarrelGeometry,
						  ct.id().rawId() );
		  pfrhCleaned->setRescale(rescaleFactor);
		  energy *= rescaleFactor;
		}
		pfrh = createHcalRecHit( detid, 
					 energy, 
				 PFLayer::HCAL_BARREL1, 
					 hcalBarrelGeometry,
					 ct.id().rawId() );
		pfrh->setRescale(rescaleFactor);
*/
	      }
	      break;
	    case HcalEndcap:
	      {
		if(energy < thresh_Endcap_ ) continue;

		/*
		// Check the timing
		if ( energy > 5. ) { 
		  for(unsigned int i=0;i< hits.size();++i) {
		    if( hits[i].det() != DetId::Hcal ) continue; 
		    HcalDetId theDetId = hits[i]; 
		    typedef HBHERecHitCollection::const_iterator iHBHE;
		    iHBHE theHit = hbheHandle->find(theDetId); 
		    if ( theHit != hbheHandle->end() ) 
		      std::cout << "HCAL hit : " 
				<< theDetId.ieta() << " " << theDetId.iphi() << " " 
				<< theHit->energy() << " " << theHit->time() << std::endl;
		  }
		}
		*/

/*
		// Apply tower 29 calibration
		if ( HCAL_Calib_ && abs(detid.ieta()) == 29 ) energy *= HCAL_Calib_29;
		//if ( rescaleFactor > 1. ) 
		// std::cout << "End-cap HCAL energy rescaled from = " << energy << " to " << energy*rescaleFactor << std::endl;
		if ( rescaleFactor > 1. ) { 
		  pfrhCleaned = createHcalRecHit( detid, 
						  energy, 
						  PFLayer::HCAL_ENDCAP, 
						  hcalEndcapGeometry,
						  ct.id().rawId() );
		  pfrhCleaned->setRescale(rescaleFactor);
		  energy *= rescaleFactor;
		}
		pfrh = createHcalRecHit( detid, 
					 energy, 
					 PFLayer::HCAL_ENDCAP, 
					 hcalEndcapGeometry,
					 ct.id().rawId() );
		pfrh->setRescale(rescaleFactor);
*/
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
		  theLongHitEnergy = theLongHit->energy();
		  flagLongDPG = applyLongShortDPG_ && theLongHit->flagField(HcalCaloFlagLabels::HFLongShort)==1;
		  flagLongTimeDPG = applyTimeDPG_ && theLongHit->flagField(HcalCaloFlagLabels::HFInTimeWindow)==1;
		  flagLongPulseDPG = applyPulseDPG_ && theLongHit->flagField(HcalCaloFlagLabels::HFDigiTime)==1;
		}
		//
		if ( theShortHit != hfHandle->end() ) { 
		  theShortHitEnergy = theShortHit->energy();
		  flagShortDPG =  applyLongShortDPG_ && theShortHit->flagField(HcalCaloFlagLabels::HFLongShort)==1;
		  flagShortTimeDPG = applyTimeDPG_ && theShortHit->flagField(HcalCaloFlagLabels::HFInTimeWindow)==1;
		  flagShortPulseDPG = applyPulseDPG_ && theShortHit->flagField(HcalCaloFlagLabels::HFDigiTime)==1;
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
						       ct.id().rawId() );
		  pfrhHFHADCleaned->setRescale(theShortHit->time());
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
						      ct.id().rawId() );
		  pfrhHFEMCleaned->setRescale(theLongHit->time());
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
		  // The channel is killed
		  if ( !theStatusValue ) { 
		    // rescaleFactor = 0. ;
		    pfrhHFHADCleaned = createHcalRecHit( detid, 
							 theShortHitEnergy, 
							 PFLayer::HF_HAD, 
							 hcalEndcapGeometry,
							 ct.id().rawId() );
		    pfrhHFHADCleaned->setRescale(theShortHit->time());
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
		  // The channel is killed
		  if ( !theStatusValue ) { 
		    //rescaleFactor = 0. ;
		    pfrhHFEMCleaned = createHcalRecHit( detid, 
						      theLongHitEnergy, 
						      PFLayer::HF_EM, 
						      hcalEndcapGeometry,
						      ct.id().rawId() );
		    pfrhHFEMCleaned->setRescale(theLongHit->time());
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
							  ct.id().rawId() );
		    pfrhHFEMCleaned29->setRescale(theLongHit->time());
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
							   ct.id().rawId() );
		    pfrhHFHADCleaned29->setRescale(theShortHit->time());
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
		    theLongHitEnergy29 = theLongHit29->energy() ;
		    flagLongDPG29 = applyLongShortDPG_ && theLongHit29->flagField(HcalCaloFlagLabels::HFLongShort)==1;
		    flagLongTimeDPG29 = applyTimeDPG_ && theLongHit29->flagField(HcalCaloFlagLabels::HFInTimeWindow)==1;
		    flagLongPulseDPG29 = applyPulseDPG_ && theLongHit29->flagField(HcalCaloFlagLabels::HFDigiTime)==1;
		  }
		  //
		  if ( theShortHit29 != hfHandle->end() ) { 		    
		    theShortHitEnergy29 = theShortHit29->energy();		  
		    flagShortDPG29 = applyLongShortDPG_ && theShortHit29->flagField(HcalCaloFlagLabels::HFLongShort)==1;
		    flagShortTimeDPG29 = applyTimeDPG_ && theShortHit29->flagField(HcalCaloFlagLabels::HFInTimeWindow)==1;
		    flagShortPulseDPG29 = applyPulseDPG_ && theShortHit29->flagField(HcalCaloFlagLabels::HFDigiTime)==1;
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
							  ct.id().rawId() );
		    pfrhHFEMCleaned29->setRescale(theLongHit29->time());
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
							 ct.id().rawId() );
		    pfrhHFHADCleaned29->setRescale(theShortHit29->time());
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
		    // The channel is killed
		    if ( !theStatusValue ) { 
		      //rescaleFactor = 0. ;
		      pfrhHFHADCleaned29 = createHcalRecHit( detid, 
							   theShortHitEnergy29, 
							   PFLayer::HF_HAD, 
							   hcalEndcapGeometry,
							   ct.id().rawId() );
		      pfrhHFHADCleaned29->setRescale(theShortHit29->time());
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
		    // The channel is killed
		    if ( !theStatusValue ) { 
		      //rescaleFactor = 0. ;
		      pfrhHFEMCleaned29 = createHcalRecHit( detid, 
							    theLongHitEnergy29, 
							    PFLayer::HF_EM, 
							    hcalEndcapGeometry,
							    ct.id().rawId() );
		      pfrhHFEMCleaned29->setRescale(theLongHit29->time());
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
							  ct.id().rawId() );
		    pfrhHFEMCleaned29->setRescale(theLongHit29->time());
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
							   ct.id().rawId() );
		    pfrhHFHADCleaned29->setRescale(theShortHit29->time());
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
					       ct.id().rawId() );
		  pfrhHFHAD = createHcalRecHit( detid, 
						energyhadHF, 
						PFLayer::HF_HAD, 
						hcalEndcapGeometry,
						ct.id().rawId() );
		  pfrhHFEM->setEnergyUp(energyhadHF);
		  pfrhHFHAD->setEnergyUp(energyemHF);
		}
		
	      }
	      break;
	    default:
	      LogError("PFHCALDualTimeRecHitProducer")
		<<"CaloTower constituent: unknown layer : "
		<<detid.subdet()<<endl;
	    } 
/*
	    if(pfrh) { 
	      rechits.push_back( *pfrh );
	      delete pfrh;
	      idSortedRecHits.insert( make_pair(ct.id().rawId(), 
						rechits.size()-1 ) ); 
	    }
	    if(pfrhCleaned) { 
	      rechitsCleaned.push_back( *pfrhCleaned );
	      delete pfrhCleaned;
	    }
*/
	    //---ab: 2 rechits for HF:	   
	    if(pfrhHFEM) { 
	      HFEMRecHits->push_back( *pfrhHFEM );
	      delete pfrhHFEM;
	      idSortedRecHitsHFEM.insert( make_pair(ct.id().rawId(), 
						HFEMRecHits->size()-1 ) ); 
	    }
	    if(pfrhHFHAD) { 
	      HFHADRecHits->push_back( *pfrhHFHAD );
	      delete pfrhHFHAD;
	      idSortedRecHitsHFHAD.insert( make_pair(ct.id().rawId(), 
						HFHADRecHits->size()-1 ) ); 
	    }
	    //---ab	   
	    if(pfrhHFEMCleaned) { 
	      rechitsCleaned.push_back( *pfrhHFEMCleaned );
	      delete pfrhHFEMCleaned;
	    }
	    if(pfrhHFHADCleaned) { 
	      rechitsCleaned.push_back( *pfrhHFHADCleaned );
	      delete pfrhHFHADCleaned;
	    }
	    if(pfrhHFEMCleaned29) { 
	      rechitsCleaned.push_back( *pfrhHFEMCleaned29 );
	      delete pfrhHFEMCleaned29;
	    }
	    if(pfrhHFHADCleaned29) { 
	      rechitsCleaned.push_back( *pfrhHFHADCleaned29 );
	      delete pfrhHFHADCleaned29;
	    }
	  }
      }
      // do navigation 
/*
      for(unsigned i=0; i<rechits.size(); i++ ) {
	findRecHitNeighboursCT( rechits[i], 
				idSortedRecHits, 
				*caloTowerTopology);
      }
*/

      for(unsigned i=0; i<rechits.size(); i++ ) {

        findRecHitNeighbours( rechits[i], idSortedRecHits,
                              *hcalTopology,
                              *hcalBarrelGeometry,
                              *hcalTopology,
                              *hcalEndcapGeometry);
      } // loop for navigation

      for(unsigned i=0; i<HFEMRecHits->size(); i++ ) {
	findRecHitNeighboursCT( (*HFEMRecHits)[i], 
				idSortedRecHitsHFEM, 
				*caloTowerTopology);
      }
      for(unsigned i=0; i<HFHADRecHits->size(); i++ ) {
	findRecHitNeighboursCT( (*HFHADRecHits)[i], 
				idSortedRecHitsHFHAD, 
				*caloTowerTopology);
      }
      iEvent.put( HFHADRecHits,"HFHAD" );	
      iEvent.put( HFEMRecHits,"HFEM" );	
    }   
  }
  else if( !(inputTagHcalRecHitsHBHE_ == InputTag()) ) { 
    // clustering is not done on CaloTowers but on HCAL rechits.

    
    // HCAL rechits 
    //    vector<edm::Handle<HBHERecHitCollection> > hcalHandles;  
    edm::Handle<HBHERecHitCollection>  hcalHandle;  
  

    
    bool found = iEvent.getByLabel(inputTagHcalRecHitsHBHE_, 
				   hcalHandle );

    if(!found) {
      ostringstream err;
      err<<"could not find rechits "<<inputTagHcalRecHitsHBHE_;
      LogError("PFHCALDualTimeRecHitProducer")<<err.str()<<endl;
    
      throw cms::Exception( "MissingProduct", err.str());
    }
    else {
      assert( hcalHandle.isValid() );
      
      const edm::Handle<HBHERecHitCollection>& handle = hcalHandle;

      for(unsigned irechit=0; irechit<handle->size(); irechit++) {
	const HBHERecHit& hit = (*handle)[irechit];

	
	double energy = hit.energy();
	
	reco::PFRecHit* pfrh = 0;
	

	const HcalDetId& detid = hit.detid();
	switch( detid.subdet() ) {
	case HcalBarrel:
	  {
	    if(energy < thresh_Barrel_ ) continue;
	    pfrh = createHcalRecHit( detid, 
				     energy, 
				     PFLayer::HCAL_BARREL1, 
				     hcalBarrelGeometry );
 	  }
	  break;
	case HcalEndcap:
	  {
	    if(energy < thresh_Endcap_ ) continue;
	    pfrh = createHcalRecHit( detid, 
				     energy, 
				     PFLayer::HCAL_ENDCAP, 
				     hcalEndcapGeometry );	  
 	  }
	  break;
	case HcalForward:
	  {
	    if(energy < thresh_HF_ ) continue;
	    pfrh = createHcalRecHit( detid, 
				     energy, 
				     PFLayer::HF_HAD, 
				     hcalEndcapGeometry );
 	  }
	  break;
	default:
	  LogError("PFHCALDualTimeRecHitProducer")
	    <<"HCAL rechit: unknown layer : "<<detid.subdet()<<endl;
	  continue;
	} 

	if(pfrh) { 
	  rechits.push_back( *pfrh );
	  delete pfrh;
	  idSortedRecHits.insert( make_pair(detid.rawId(), 
					    rechits.size()-1 ) ); 
	}
      }
      
      
      // do navigation:
      for(unsigned i=0; i<rechits.size(); i++ ) {
	
	findRecHitNeighbours( rechits[i], idSortedRecHits, 
			      *hcalTopology, 
			      *hcalBarrelGeometry, 
			      *hcalTopology,
			      *hcalEndcapGeometry);
      } // loop for navigation
    }  // endif hcal rechits were found
  } // endif clustering on rechits in hcal
}






reco::PFRecHit* 
PFHCALDualTimeRecHitProducer::createHcalRecHit( const DetId& detid,
					double energy,
					PFLayer::Layer layer,
					const CaloSubdetectorGeometry* geom,
					unsigned newDetId ) {
  
  const CaloCellGeometry *thisCell = geom->getGeometry(detid);
  if(!thisCell) {
    edm::LogError("PFHCALDualTimeRecHitProducer")
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

  unsigned id = detid;
  if(newDetId) id = newDetId;
  reco::PFRecHit *rh = 
    new reco::PFRecHit( id,  layer, energy, 
			position.x(), position.y(), position.z()+depth_correction, 
			0,0,0 );
 
  
  
  
  // set the corners
  const CaloCellGeometry::CornersVec& corners = thisCell->getCorners();

  assert( corners.size() == 8 );

  rh->setNECorner( corners[0].x(), corners[0].y(),  corners[0].z()+depth_correction );
  rh->setSECorner( corners[1].x(), corners[1].y(),  corners[1].z()+depth_correction );
  rh->setSWCorner( corners[2].x(), corners[2].y(),  corners[2].z()+depth_correction );
  rh->setNWCorner( corners[3].x(), corners[3].y(),  corners[3].z()+depth_correction );
 
  return rh;
}




void 
PFHCALDualTimeRecHitProducer::findRecHitNeighbours
( reco::PFRecHit& rh, 
  const map<unsigned,unsigned >& sortedHits, 
  const CaloSubdetectorTopology& barrelTopology, 
  const CaloSubdetectorGeometry& barrelGeometry, 
  const CaloSubdetectorTopology& endcapTopology, 
  const CaloSubdetectorGeometry& endcapGeometry ) {
  
  //cout<<"------PFRecHitProducerHcaL:findRecHitNeighbours navigation value "<<navigation_HF_<<endl;
 if(navigation_HF_ == false){
    if( rh.layer() == PFLayer::HF_HAD )
      return;
    if( rh.layer() == PFLayer::HF_EM )
      return;
  } 
  DetId detid( rh.detId() );

  const CaloSubdetectorTopology* topology = 0;
  const CaloSubdetectorGeometry* geometry = 0;
  // const CaloSubdetectorGeometry* othergeometry = 0;
  
  switch( rh.layer() ) {
  case PFLayer::ECAL_ENDCAP: 
    topology = &endcapTopology;
    geometry = &endcapGeometry;
    break;
  case PFLayer::ECAL_BARREL: 
    topology = &barrelTopology;
    geometry = &barrelGeometry;
    break;
  case PFLayer::HCAL_ENDCAP:
    topology = &endcapTopology;
    geometry = &endcapGeometry;
    // othergeometry = &barrelGeometry;
    break;
  case PFLayer::HCAL_BARREL1:
    topology = &barrelTopology;
    geometry = &barrelGeometry;
    // othergeometry = &endcapGeometry;
    break;
  case PFLayer::PS1:
  case PFLayer::PS2:
    topology = &barrelTopology;
    geometry = &barrelGeometry;
    // othergeometry = &endcapGeometry;
    break;
  default:
    assert(0);
  }
  
  assert( topology && geometry );

  CaloNavigator<DetId> navigator(detid, topology);

  DetId north = navigator.north();  
  
  DetId northeast(0);
  if( north != DetId(0) ) {
    northeast = navigator.east();  
  }
  navigator.home();


  DetId south = navigator.south();

  

  DetId southwest(0); 
  if( south != DetId(0) ) {
    southwest = navigator.west();
  }
  navigator.home();


  DetId east = navigator.east();
  DetId southeast;
  if( east != DetId(0) ) {
    southeast = navigator.south(); 
  }
  navigator.home();
  DetId west = navigator.west();
  DetId northwest;
  if( west != DetId(0) ) {   
    northwest = navigator.north();  
  }
  navigator.home();
    
  IDH i = sortedHits.find( north.rawId() );
  if(i != sortedHits.end() ) 
    rh.add4Neighbour( i->second );
  
  i = sortedHits.find( northeast.rawId() );
  if(i != sortedHits.end() ) 
    rh.add8Neighbour( i->second );
  
  i = sortedHits.find( south.rawId() );
  if(i != sortedHits.end() ) 
    rh.add4Neighbour( i->second );
    
  i = sortedHits.find( southwest.rawId() );
  if(i != sortedHits.end() ) 
    rh.add8Neighbour( i->second );
    
  i = sortedHits.find( east.rawId() );
  if(i != sortedHits.end() ) 
    rh.add4Neighbour( i->second );
    
  i = sortedHits.find( southeast.rawId() );
  if(i != sortedHits.end() ) 
    rh.add8Neighbour( i->second );
    
  i = sortedHits.find( west.rawId() );
  if(i != sortedHits.end() ) 
     rh.add4Neighbour( i->second );
   
  i = sortedHits.find( northwest.rawId() );
  if(i != sortedHits.end() ) 
    rh.add8Neighbour( i->second );
    

}


void 
PFHCALDualTimeRecHitProducer::findRecHitNeighboursCT
( reco::PFRecHit& rh, 
  const map<unsigned, unsigned >& sortedHits, 
  const CaloSubdetectorTopology& topology ) {
  //cout<<"------PFRecHitProducerHcaL:findRecHitNeighboursCT navigation value "<<navigation_HF_<<endl;
  //  cout<<"----------- rechit print out"<<endl;
  // if(( rh.layer() == PFLayer::HF_HAD )||(rh.layer() == PFLayer::HF_EM)) {  
    
  //    cout<<rh<<endl;
    //  }
  if(navigation_HF_ == false){
    if( rh.layer() == PFLayer::HF_HAD )
      return;
    if( rh.layer() == PFLayer::HF_EM )
      return;
  }
  CaloTowerDetId ctDetId( rh.detId() );
    

  vector<DetId> northids = topology.north(ctDetId);
  vector<DetId> westids = topology.west(ctDetId);
  vector<DetId> southids = topology.south(ctDetId);
  vector<DetId> eastids = topology.east(ctDetId);


  CaloTowerDetId badId;

  // all the following detids will be CaloTowerDetId
  CaloTowerDetId north;
  CaloTowerDetId northwest;
  CaloTowerDetId northwest2;
  CaloTowerDetId west;
  CaloTowerDetId west2;
  CaloTowerDetId southwest;
  CaloTowerDetId southwest2;
  CaloTowerDetId south;
  CaloTowerDetId southeast;
  CaloTowerDetId southeast2;
  CaloTowerDetId east;
  CaloTowerDetId east2;
  CaloTowerDetId northeast;
  CaloTowerDetId northeast2;
  
  // for north and south, there is no ambiguity : 1 or 0 neighbours
  
  switch( northids.size() ) {
  case 0: 
    break;
  case 1: 
    north = northids[0];
    break;
  default:
  stringstream err("PFHCALDualTimeRecHitProducer::findRecHitNeighboursCT : incorrect number of neighbours north: "); 
    err<<northids.size();
    throw( err.str() ); 
  }

  switch( southids.size() ) {
  case 0: 
    break;
  case 1: 
    south = southids[0];
    break;
  default:
  stringstream err("PFHCALDualTimeRecHitProducer::findRecHitNeighboursCT : incorrect number of neighbours south: "); 
    err<<southids.size();
    throw( err.str() ); 
  }
  
  // for east and west, one must take care 
  // of the pitch change in HCAL endcap.

  switch( eastids.size() ) {
  case 0: 
    break;
  case 1: 
    east = eastids[0];
    northeast = getNorth(east, topology);
    southeast = getSouth(east, topology);
    break;
  case 2:  
    // in this case, 0 is more on the north than 1
    east = eastids[0];
    east2 = eastids[1];
    northeast = getNorth(east, topology );
    southeast = getSouth(east2, topology);    
    northeast2 = getNorth(northeast, topology );
    southeast2 = getSouth(southeast, topology);    
    break;
  default:
  stringstream err("PFHCALDualTimeRecHitProducer::findRecHitNeighboursCT : incorrect number of neighbours eastids: "); 
    err<<eastids.size();
    throw( err.str() ); 
  }
  
  
  switch( westids.size() ) {
  case 0: 
    break;
  case 1: 
    west = westids[0];
    northwest = getNorth(west, topology);
    southwest = getSouth(west, topology);
    break;
  case 2:  
    // in this case, 0 is more on the north than 1
    west = westids[0];
    west2 = westids[1];
    northwest = getNorth(west, topology );
    southwest = getSouth(west2, topology );    
    northwest2 = getNorth(northwest, topology );
    southwest2 = getSouth(southwest, topology );    
    break;
  default:
  stringstream err("PFHCALDualTimeRecHitProducer::findRecHitNeighboursCT : incorrect number of neighbours westids: "); 
    err<< westids.size();
    throw( err.str() ); 
  }




  // find and set neighbours
    
  IDH i = sortedHits.find( north.rawId() );
  if(i != sortedHits.end() ) 
    rh.add4Neighbour( i->second );
  
  i = sortedHits.find( northeast.rawId() );
  if(i != sortedHits.end() ) 
    rh.add8Neighbour( i->second );
  
  i = sortedHits.find( northeast2.rawId() );
  if(i != sortedHits.end() ) 
    rh.add8Neighbour( i->second );
  
  i = sortedHits.find( south.rawId() );
  if(i != sortedHits.end() ) 
    rh.add4Neighbour( i->second );
    
  i = sortedHits.find( southwest.rawId() );
  if(i != sortedHits.end() ) 
    rh.add8Neighbour( i->second );
    
  i = sortedHits.find( southwest2.rawId() );
  if(i != sortedHits.end() ) 
    rh.add8Neighbour( i->second );
    
  i = sortedHits.find( east.rawId() );
  if(i != sortedHits.end() ) 
    rh.add4Neighbour( i->second );
    
  i = sortedHits.find( east2.rawId() );
  if(i != sortedHits.end() ) 
    rh.add4Neighbour( i->second );
    
  i = sortedHits.find( southeast.rawId() );
  if(i != sortedHits.end() ) 
    rh.add8Neighbour( i->second );
    
  i = sortedHits.find( southeast2.rawId() );
  if(i != sortedHits.end() ) 
    rh.add8Neighbour( i->second );
    
  i = sortedHits.find( west.rawId() );
  if(i != sortedHits.end() ) 
     rh.add4Neighbour( i->second );
   
  i = sortedHits.find( west2.rawId() );
  if(i != sortedHits.end() ) 
     rh.add4Neighbour( i->second );
   
  i = sortedHits.find( northwest.rawId() );
  if(i != sortedHits.end() ) 
    rh.add8Neighbour( i->second );

  i = sortedHits.find( northwest2.rawId() );
  if(i != sortedHits.end() ) 
    rh.add8Neighbour( i->second );

  //  cout<<"----------- rechit print out"<<endl;
  // if(( rh.layer() == PFLayer::HF_HAD )||(rh.layer() == PFLayer::HF_EM)) {  
    
  //   cout<<rh<<endl;
    //  }
}



DetId 
PFHCALDualTimeRecHitProducer::getSouth(const DetId& id, 
			       const CaloSubdetectorTopology& topology) {

  DetId south;
  vector<DetId> sids = topology.south(id);
  if(sids.size() == 1)
    south = sids[0];
  
  return south;
} 



DetId 
PFHCALDualTimeRecHitProducer::getNorth(const DetId& id, 
			       const CaloSubdetectorTopology& topology) {

  DetId north;
  vector<DetId> nids = topology.north(id);
  if(nids.size() == 1)
    north = nids[0];
  
  return north;
} 



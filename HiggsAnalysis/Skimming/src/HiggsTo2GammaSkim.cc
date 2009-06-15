
/* \class HiggsTo2GammaSkim 
 *
 * Consult header file for description
 *
 * author:  Kati Lassila-Perini Helsinki Institute of Physics
 *
 */


// system include files
#include <HiggsAnalysis/Skimming/interface/HiggsTo2GammaSkim.h>

// User include files
#include <FWCore/ParameterSet/interface/ParameterSet.h>

// Message logger
#include <FWCore/MessageLogger/interface/MessageLogger.h>

// Photons:
#include <DataFormats/EgammaCandidates/interface/Photon.h>
#include <DataFormats/EgammaCandidates/interface/PhotonFwd.h>

// not for 3_1 //#include "DataFormats/EgammaCandidates/interface/PhotonIDFwd.h"
// not for 3_1 //#include "DataFormats/EgammaCandidates/interface/PhotonID.h"
// not for 3_1 //#include "DataFormats/EgammaCandidates/interface/PhotonIDAssociation.h"

#include "FWCore/Framework/interface/ESHandle.h"

// C++
#include <iostream>
#include <vector>
#include <map>

using namespace std;
using namespace edm;
using namespace reco;


// Constructor
HiggsTo2GammaSkim::HiggsTo2GammaSkim(const edm::ParameterSet& pset) {

  /*
  outputFile_   = pset.getParameter<std::string>("outputFile");
  // Read variables that must be passed to allow a 
  //  supercluster to be placed in histograms as a photon.
  minPhotonEt_     = pset.getParameter<double>("minPhotonEt");
  minPhotonAbsEta_ = pset.getParameter<double>("minPhotonAbsEta");
  maxPhotonAbsEta_ = pset.getParameter<double>("maxPhotonAbsEta");
  minPhotonR9_     = pset.getParameter<double>("minPhotonR9");
  maxPhotonHoverE_ = pset.getParameter<double>("maxPhotonHoverE");

  // Read variable to that decidedes whether
  // a TTree of photons is created or not
  createPhotonTTree_ = pset.getParameter<bool>("createPhotonTTree");

  // open output file to store histograms
  rootFile_ = TFile::Open(outputFile_.c_str(),"RECREATE");
  */

  // Local Debug flag
  debug              = pset.getParameter<bool>("DebugHiggsTo2GammaSkim");

  // Reconstructed objects
  thePhotonLabel     = pset.getParameter<edm::InputTag>("PhotonCollectionLabel");



  // Minimum Pt for photons for skimming
  //photon1MinPt       = pset.getParameter<double>("photon1MinimumPt");
  //nPhotonMin         = pset.getParameter<int>("nPhotonMinimum");

   photonLooseMinPt = pset.getParameter<double>("photonLooseMinPt");
   photonTightMinPt = pset.getParameter<double>("photonTightMinPt");
   photonLooseMaxEta = pset.getParameter<double>("photonLooseMaxEta");
   photonTightMaxEta = pset.getParameter<double>("photonTightMaxEta");
   photonLooseMaxHoE = pset.getParameter<double>("photonLooseMaxHoE");
   photonTightMaxHoE = pset.getParameter<double>("photonTightMaxHoE");
   photonLooseMaxHIsol = pset.getParameter<double>("photonLooseMaxHIsol");
   photonTightMaxHIsol = pset.getParameter<double>("photonTightMaxHIsol");
   photonLooseMaxEIsol = pset.getParameter<double>("photonLooseMaxEIsol");
   photonTightMaxEIsol = pset.getParameter<double>("photonTightMaxEIsol");
   photonLooseMaxTIsol = pset.getParameter<double>("photonLooseMaxTIsol");
   photonTightMaxTIsol = pset.getParameter<double>("photonTightMaxTIsol");

  //float photon2MinPt;

   nPhotonLooseMin = pset.getParameter<int>("nPhotonLooseMin"); //includes tight
   nPhotonTightMin = pset.getParameter<int>("nPhotonTightMin");

   nEvents         = 0;
   nSelectedEvents = 0;
   n1loose = 0;
   n1tight = 0;
   n2loose = 0;
   n2tight = 0;
   n1loose1tight = 0;
   n0loosetight = 0;

}


// Destructor
HiggsTo2GammaSkim::~HiggsTo2GammaSkim() {

  edm::LogVerbatim("HiggsTo2GammaSkim") 
    << " Number_events_read " << nEvents          
    << " Number_events_kept " << nSelectedEvents 
    << " Efficiency         " << ((double)nSelectedEvents)/((double) nEvents + 0.01) 
    << " Number_events_two_photon_tight " <<  n2tight         
    << " Number_events_one_photon_tight_one_photon_loose " <<  n1loose1tight         
    << " Number_events_two_photon_loose " <<  n2loose         
    << " Number_events_one_photon_tight " <<  n1tight         
    << " Number_events_one_photon_loose " <<  n1loose         
    << " Number_events_no_photon " <<  n0loosetight         
    << std::endl;

  if(debug) {
      std::cout << " Number_events_read                              " << nEvents   << std::endl;       
      std::cout << " Number_events_kept                              " << nSelectedEvents << std::endl;
      std::cout << " Efficiency                                      " << ((double)nSelectedEvents)/((double) nEvents + 0.01) << std::endl;
      std::cout << " Number_events_two_photon_tight                  " <<  n2tight   << std::endl;      
      std::cout << " Number_events_one_photon_tight_one_photon_loose " <<  n1loose1tight << std::endl;        
      std::cout << " Number_events_two_photon_loose                  " <<  n2loose   << std::endl;      
      std::cout << " Number_events_one_photon_tight                  " <<  n1tight    << std::endl;     
      std::cout << " Number_events_one_photon_loose                  " <<  n1loose   << std::endl;      
      std::cout << " Number_events_no_photon                         " <<  n0loosetight   << std::endl;      
  }

}


// Filter event
bool HiggsTo2GammaSkim::filter(edm::Event& event, const edm::EventSetup& setup ) {
  
  nEvents++;
  
  using reco::PhotonCollection;
  //using reco::PhotonIDAssociationCollection;

  bool keepEvent    = false;
  int  nPhotonsLoose = 0;
  int nPhotonsTight = 0;
  
  // Look at photons:

  // Get the photon collection from the event
  //edm::Handle<reco::PhotonCollection> photonHandle;
  //event.getByLabel(thePhotonLabel.label(),photonHandle);

  // grab photons
  Handle<reco::PhotonCollection> photonHandle;
  event.getByLabel("photons", "", photonHandle);

  //ADDED
  // grab PhotonId objects  
  // not for 3_1 //edm::Handle<reco::PhotonIDAssociationCollection> photonIDMapColl;
  // not for 3_1 //event.getByLabel("PhotonIDProd", "PhotonAssociatedID", photonIDMapColl);

  const reco::PhotonCollection* phoCollection = photonHandle.product();
  //reco::PhotonCollection::const_iterator photons;
  // not for 3_1 //const reco::PhotonIDAssociationCollection *phoMap = photonIDMapColl.product();

  if ( photonHandle.isValid() ) {
  
    // Loop over photon collections and count how many photons there are, 
    // and how many are above the thresholds

    // 1) Primary Vertex
    // 2) Sigmaetaeta cut
    // 3) see if using SC corrected or raw energy and also SC collection for loose?!?! (commented later)
    // 4) check definition of isolation and hoe, maybe decouple from photon id cfi

    for (int i=0; i<int(phoCollection->size()); i++) {   
      
      edm::Ref<reco::PhotonCollection> photon(photonHandle, i);
      
      // not for 3_1 //reco::PhotonIDAssociationCollection::const_iterator photonIter = phoMap->find(photon);
      //const reco::PhotonRef &pho = photonIter->key;
      // not for 3_1 //const reco::PhotonIDRef &photonId = photonIter->val;

      float photonEt       = photon->et();
      //float superClusterEt = (photon->superCluster()->energy())/(cosh(photon->superCluster()->position().eta()));
      //float superClusterEta = photon->superCluster()->eta();

      bool passCutsLoose = ( photonEt >photonLooseMinPt ) && 
	( fabs(photon->eta()) < photonLooseMaxEta ) &&
	( photon->hadronicOverEm() < photonLooseMaxHoE  || photonLooseMaxHoE < 0. )  &&

	//Change for 3_1 //((photonId)->isolationEcalRecHit() < photonLooseMaxEIsol || photonLooseMaxEIsol<0.) &&
	//Change for 3_1 //((photonId)->isolationHcalRecHit() < photonLooseMaxHIsol || photonLooseMaxHIsol<0.) &&
	//Change for 3_1 //((photonId)->isolationHollowTrkCone() < photonLooseMaxTIsol || photonLooseMaxTIsol<0.) 
	(photon->ecalRecHitSumEtConeDR03() < photonLooseMaxEIsol || photonLooseMaxEIsol<0.) &&
	(photon->hcalTowerSumEtConeDR03() < photonLooseMaxHIsol || photonLooseMaxHIsol<0.) &&
	(photon->trkSumPtHollowConeDR03() < photonLooseMaxTIsol || photonLooseMaxTIsol<0.) 
	;

      bool passCutsTight = ( photonEt >photonTightMinPt ) && 
	( fabs(photon->eta()) < photonTightMaxEta ) &&
	( photon->hadronicOverEm() < photonTightMaxHoE  || photonTightMaxHoE < 0. )  &&
	//Change for 3_1 //((photonId)->isolationEcalRecHit() < photonTightMaxEIsol || photonTightMaxEIsol<0.) &&
	//Change for 3_1 //((photonId)->isolationHcalRecHit() < photonTightMaxHIsol || photonTightMaxHIsol<0.) &&
	//Change for 3_1 //((photonId)->isolationHollowTrkCone() < photonTightMaxTIsol || photonTightMaxTIsol<0.) 
	(photon->ecalRecHitSumEtConeDR03() < photonTightMaxEIsol || photonTightMaxEIsol<0.) &&
	(photon->hcalTowerSumEtConeDR03() < photonTightMaxHIsol || photonTightMaxHIsol<0.) &&
	(photon->trkSumPtHollowConeDR03() < photonTightMaxTIsol || photonTightMaxTIsol<0.) 
	;


      if ( passCutsTight ) {
	nPhotonsTight++;
      }
      if ( passCutsLoose ) {
	nPhotonsLoose++;
      }

      if(debug) {
	if ( passCutsTight ) {
	  cout<<"Tight ";
	  cout<<"photon: et, eta, hoe, isoe, isoh, isot "<<photon->et()<<" "<<photon->eta()<<" "<< photon->hadronicOverEm()<<" "
	    //Change for 3_1 //<<(photonId)->isolationEcalRecHit()<<" "<<(photonId)->isolationHcalRecHit()<<" "<<(photonId)->isolationHollowTrkCone()<<endl;
	      <<photon->ecalRecHitSumEtConeDR03()<<" "<<photon->hcalTowerSumEtConeDR03()<<" "<<photon->trkSumPtHollowConeDR03()<<endl;
	}
	else {
	  if ( passCutsLoose ) {
	    cout<<"Loose ";
	    cout<<"photon: et, eta, hoe, isoe, isoh, isot "<<photon->et()<<" "<<photon->eta()<<" "<< photon->hadronicOverEm()<<" "
	      //Change for 3_1 //<<(photonId)->isolationEcalRecHit()<<" "<<(photonId)->isolationHcalRecHit()<<" "<<(photonId)->isolationHollowTrkCone()<<endl;
		<<photon->ecalRecHitSumEtConeDR03()<<" "<<photon->hcalTowerSumEtConeDR03()<<" "<<photon->trkSumPtHollowConeDR03()<<endl;
	  }
	}
      }
    }
  }
  
  // Make decision:
  if ( nPhotonsLoose >= nPhotonLooseMin && nPhotonsTight >= nPhotonTightMin) keepEvent = true;

  if (keepEvent) nSelectedEvents++;
  if(debug) {
    cout<<"selection: nPhotonLooseMin nPhotonTightMin photonLooseMinPt photonTightMinPt nLoose, nTight "<<nPhotonLooseMin<<" "<<photonLooseMinPt<<" "<<nPhotonTightMin<<" "<<photonTightMinPt<<" "<<nPhotonsLoose<<" "<<nPhotonsTight<<" keep "<<keepEvent<<" nSelected "<<nSelectedEvents<<endl;
  }

  if(nPhotonsTight>=2) {
    n2tight++;
  }
  else if(nPhotonsTight==1) {
    if(nPhotonsLoose>=2) {
      n1loose1tight++;
    }
    else {
      n1tight++;
    }
  }
  else if(nPhotonsLoose>=2) {
    n2loose++;
  }
  else if(nPhotonsLoose=1) {
    n1loose++;
  }
  else {
    n0loosetight++;
  }
  
  return keepEvent;
}

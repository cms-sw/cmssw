#include "DataFormats/Math/interface/LorentzVector.h"
#include "RecoMET/METAlgorithms/interface/SignCaloSpecificAlgo.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "RecoMET/METAlgorithms/interface/SigInputObj.h"
#include "RecoMET/METAlgorithms/interface/significanceAlgo.h"
#include "RecoMET/METAlgorithms/interface/SignAlgoResolutions.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


#include <string>
using namespace reco;
using namespace std;
// -*- C++ -*-
//
// Package:    METAlgorithms
// Class:      SigInputObj
// 
/**\class METSignificance SigInputObj.cc RecoMET/METAlgorithms/src/SigInputObj.cc

 Description: <one line class summary>

 Implementation:

-------------------------------------------------------------------------
 This algorithm adds calorimeter specific global event information to 
 the MET object which may be useful/needed for MET Data Quality Monitoring
 and MET cleaning.  This list is not exhaustive and additional 
 information will be added in the future. 
-------------------------------------

*/
//
// Original Author:  Kyle Story, Freya Blekman (Cornell University)
//         Created:  Fri Apr 18 11:58:33 CEST 2008
// $Id: SignCaloSpecificAlgo.cc,v 1.3 2008/11/07 12:10:09 fblekman Exp $
//
//
reco::CaloMET SignCaloSpecificAlgo::addInfo(edm::Handle<edm::View<Candidate> > towers, CommonMETData met, const metsig::SignAlgoResolutions & resolutions, bool noHF, double globalThreshold)
{ 
  // Instantiate the container to hold the calorimeter specific information
  SpecificCaloMETData specific;
  // Initialise the container 
  specific.MaxEtInEmTowers = 0.0;         // Maximum energy in EM towers
  specific.MaxEtInHadTowers = 0.0;        // Maximum energy in HCAL towers
  specific.HadEtInHO = 0.0;          // Hadronic energy fraction in HO
  specific.HadEtInHB = 0.0;          // Hadronic energy in HB
  specific.HadEtInHF = 0.0;          // Hadronic energy in HF
  specific.HadEtInHE = 0.0;          // Hadronic energy in HE
  specific.EmEtInEB = 0.0;           // Em energy in EB
  specific.EmEtInEE = 0.0;           // Em energy in EE
  specific.EmEtInHF = 0.0;           // Em energy in HF
  specific.EtFractionHadronic = 0.0; // Hadronic energy fraction
  specific.EtFractionEm = 0.0;       // Em energy fraction
  specific.METSignificance = -1.0;    // MET Significance
  specific.CaloSETInpHF = 0.0;        // CaloSET in HF+ 
  specific.CaloSETInmHF = 0.0;        // CaloSET in HF- 
  specific.CaloMETInpHF = 0.0;        // CaloMET in HF+ 
  specific.CaloMETInmHF = 0.0;        // CaloMET in HF- 
  specific.CaloMETPhiInpHF = 0.0;     // CaloMET-phi in HF+ 
  specific.CaloMETPhiInmHF = 0.0;     // CaloMET-phi in HF- 
  
  double totalEt = 0.0; 
  double totalEm     = 0.0;
  double totalHad    = 0.0;
  double MaxTowerEm  = 0.0;
  double MaxTowerHad = 0.0;
  double sumEtInpHF = 0.0;
  double sumEtInmHF = 0.0;
  double MExInpHF = 0.0;
  double MEyInpHF = 0.0;
  double MExInmHF = 0.0;
  double MEyInmHF = 0.0;

  //  std::cout << "number of towers = " << towers->size() << std::endl;
  if( towers->size() == 0 )  // if there are no towers, return specific = 0
    {
      cout << "[CaloMET] Number of Candidate CaloTowers is zero : Unable to calculate calo specific info. " << endl;
      const LorentzVector p4( met.mex, met.mey, 0.0, met.met );
      const Point vtx( 0.0, 0.0, 0.0 );
      CaloMET specificmet( specific, met.sumet, p4, vtx );
      return specificmet;
    }
  //retreive calo tower information from candidates
  //start with the first element of the candidate list

  edm::View<Candidate>::const_iterator towerCand = towers->begin();

  // use this container to calculate the significance. SigInputObj are objects that contain both directional and uncertainty information and are used as input to the significance calculation

  std::vector<metsig::SigInputObj> signInputVec;
  
  //  std::cout << "starting loop over towers..." << std::endl;
  //iterate over all CaloTowers and record information
  for( ; towerCand != towers->end(); towerCand++ )  {
    const Candidate *candidate = &(*towerCand);
    if(candidate){
      const CaloTower * calotower = dynamic_cast<const CaloTower*> (candidate);
      if(calotower){
	if(calotower->et()<globalThreshold)
	  continue;
	totalEt  += calotower->et();
	totalEm  += calotower->emEt();
	totalHad += calotower->hadEt();
	bool wasused=false;
	double sign_tower_et = calotower->et();
	double sign_tower_phi = calotower->phi();
	double sign_tower_sigma_et = 0;
	double sign_tower_sigma_phi = 0;
	std::string sign_tower_type = "";
	
	bool hadIsDone = false;
	bool emIsDone = false;
	int cell = calotower->constituentsSize();
	while ( --cell >= 0 && (!hadIsDone || !emIsDone) ) 
	  {
	    
	    DetId id = calotower->constituent( cell );
	    if( !hadIsDone && id.det() == DetId::Hcal ) 
	      {
		
		HcalSubdetector subdet = HcalDetId(id).subdet();
		if(subdet == HcalBarrel){
		  sign_tower_type = "hadcalotower";
		  sign_tower_et = calotower->hadEt();
		  sign_tower_sigma_et = resolutions.eval(metsig::caloHB,metsig::ET,calotower->hadEt(),calotower->phi(),calotower->eta());
		  sign_tower_sigma_phi = resolutions.eval(metsig::caloHB,metsig::PHI,calotower->hadEt(),calotower->phi(),calotower->eta());
		}
		else if(subdet==HcalOuter){
		  sign_tower_type = "hadcalotower";
		  sign_tower_et = calotower->outerEt();
		  sign_tower_sigma_et = resolutions.eval(metsig::caloHO,metsig::ET,calotower->outerEt(),calotower->phi(),calotower->eta());
		  sign_tower_sigma_phi = resolutions.eval(metsig::caloHO,metsig::PHI,calotower->outerEt(),calotower->phi(),calotower->eta());
		}
		else if(subdet==HcalEndcap){
		  sign_tower_type = "hadcalotower";
		  sign_tower_et = calotower->hadEt();
		  sign_tower_sigma_et = resolutions.eval(metsig::caloHE,metsig::ET,calotower->hadEt(),calotower->phi(),calotower->eta());
		  sign_tower_sigma_phi = resolutions.eval(metsig::caloHE,metsig::PHI,calotower->hadEt(),calotower->phi(),calotower->eta());
		}
		else if(subdet == HcalForward){
		  sign_tower_type = "hadcalotower";
		  sign_tower_et = calotower->et();
		  sign_tower_sigma_et = resolutions.eval(metsig::caloHF,metsig::ET,calotower->et(),calotower->phi(),calotower->eta());
		  sign_tower_sigma_phi = resolutions.eval(metsig::caloHF,metsig::PHI,calotower->et(),calotower->phi(),calotower->eta());
		}
		else{
		  edm::LogWarning("SignCaloSpecificAlgo") << " HCAL tower cell not assigned to an HCAL subdetector!!!" << std::endl;
		}
		// and book!
		metsig::SigInputObj temp(sign_tower_type,sign_tower_et,sign_tower_phi,sign_tower_sigma_et,sign_tower_sigma_phi);
		if(!noHF || subdet !=HcalForward)
		  signInputVec.push_back(temp);
		
		wasused=1;
		hadIsDone = true;
	      }
	    else if( !emIsDone && id.det() == DetId::Ecal )
	      {
		EcalSubdetector subdet = EcalSubdetector( id.subdetId() );
		
		if(subdet == EcalBarrel){
		  sign_tower_type = "emcalotower";
		  sign_tower_et = calotower->emEt();
		  sign_tower_sigma_et = resolutions.eval(metsig::caloEB,metsig::ET,calotower->emEt(),calotower->phi(),calotower->eta());
		  sign_tower_sigma_phi = resolutions.eval(metsig::caloEB,metsig::PHI,calotower->emEt(),calotower->phi(),calotower->eta());
		}
		else if(subdet == EcalEndcap ){
		  sign_tower_type = "emcalotower";
		  sign_tower_et = calotower->emEt();
		  sign_tower_sigma_et = resolutions.eval(metsig::caloEE,metsig::ET,calotower->emEt(),calotower->phi(),calotower->eta());
		  sign_tower_sigma_phi = resolutions.eval(metsig::caloEE,metsig::PHI,calotower->emEt(),calotower->phi(),calotower->eta());
		  
		}
		else{
		  edm::LogWarning("SignCaloSpecificAlgo") << " ECAL tower cell not assigned to an ECAL subdetector!!!" << std::endl;
		}
		metsig::SigInputObj temp(sign_tower_type,sign_tower_et,sign_tower_phi,sign_tower_sigma_et,sign_tower_sigma_phi);
		signInputVec.push_back(temp);
		wasused=1;
		emIsDone = true;
	      }
	  }
	if(wasused==0)
	  edm::LogWarning("SignCaloSpecificAlgo") << "found non-assigned cell, " << std::endl;
      }
    }
  }

  // now run the significance algorithm.
  
  double sign_calo_met_total=0;
  double sign_calo_met_phi=0;
  double sign_calo_met_set=0;
  double significance = metsig::ASignificance(signInputVec, sign_calo_met_total, sign_calo_met_phi, sign_calo_met_set);
  met.mex = sign_calo_met_total * cos(sign_calo_met_phi);
  met.mey = sign_calo_met_total * sin(sign_calo_met_phi);
  met.sumet = sign_calo_met_set;
  met.met = sign_calo_met_total;
  met.phi = sign_calo_met_phi;

  //  std::cout << "met = " << met.met << " phi " << met.phi << " sumet " << met.sumet << " signif = " << significance << std::endl; 
  specific.METSignificance=significance;
  
  const LorentzVector p4( met.mex, met.mey, 0.0, met.met );
  const Point vtx( 0.0, 0.0, 0.0 );
  // Create and return an object of type CaloMET, which is a MET object with 
  // the extra calorimeter specfic information added
  CaloMET specificmet( specific, met.sumet, p4, vtx );
  // cleanup everything:
  signInputVec.clear();
  // and return
  return specificmet;
}
//-------------------------------------------------------------------------




double SignCaloSpecificAlgo::addSignificance(edm::Handle<edm::View<Candidate> > towers, CommonMETData met, const metsig::SignAlgoResolutions & resolutions, bool noHF, double globalThreshold)
{ 
  if( towers->size() == 0 )  // if there are no towers, return specific = 0
    {
      cout << "[CaloMET] Number of Candidate CaloTowers is zero : Unable to calculate METSignificance"  << endl;
      return 0.0;
    }
  //retreive calo tower information from candidates
  //start with the first element of the candidate list

  edm::View<Candidate>::const_iterator towerCand = towers->begin();

  // use this container to calculate the significance. SigInputObj are objects that contain both directional and uncertainty information and are used as input to the significance calculation

  std::vector<metsig::SigInputObj> signInputVec;
  
  //iterate over all CaloTowers and record information
  for( ; towerCand != towers->end(); towerCand++ ) {
    const Candidate *candidate = &(*towerCand);
    if(candidate){
      const CaloTower * calotower = dynamic_cast<const CaloTower*> (candidate);
      if(calotower){
	if(calotower->et()<globalThreshold)
	  continue;
	bool wasused=false;
	double sign_tower_et = calotower->et();
	double sign_tower_phi = calotower->phi();
	double sign_tower_sigma_et = 0;
	double sign_tower_sigma_phi = 0;
	std::string sign_tower_type = "";
	
	bool hadIsDone = false;
	bool emIsDone = false;
	int cell = calotower->constituentsSize();

	while ( --cell >= 0 && (!hadIsDone || !emIsDone) ) 
	  {
	    DetId id = calotower->constituent( cell );
	    if( !hadIsDone && id.det() == DetId::Hcal ) 
	      {
		HcalSubdetector subdet = HcalDetId(id).subdet();
		if(subdet == HcalBarrel){
		  sign_tower_type = "hadcalotower";
		  sign_tower_et = calotower->hadEt();
		  sign_tower_sigma_et = resolutions.eval(metsig::caloHB,metsig::ET,calotower->hadEt(),calotower->phi(),calotower->eta());
		  sign_tower_sigma_phi = resolutions.eval(metsig::caloHB,metsig::PHI,calotower->hadEt(),calotower->phi(),calotower->eta());
		}
		else if(subdet==HcalOuter){
		  sign_tower_type = "hadcalotower";
		  sign_tower_et = calotower->outerEt();
		  sign_tower_sigma_et = resolutions.eval(metsig::caloHO,metsig::ET,calotower->outerEt(),calotower->phi(),calotower->eta());
		  sign_tower_sigma_phi = resolutions.eval(metsig::caloHO,metsig::PHI,calotower->outerEt(),calotower->phi(),calotower->eta());
		}
		else if(subdet==HcalEndcap){
		  sign_tower_type = "hadcalotower";
		  sign_tower_et = calotower->hadEt();
		  sign_tower_sigma_et = resolutions.eval(metsig::caloHE,metsig::ET,calotower->hadEt(),calotower->phi(),calotower->eta());
		  sign_tower_sigma_phi = resolutions.eval(metsig::caloHE,metsig::PHI,calotower->hadEt(),calotower->phi(),calotower->eta());
		}
		else if(subdet == HcalForward){
		  sign_tower_type = "hadcalotower";
		  sign_tower_et = calotower->et();
		  sign_tower_sigma_et = resolutions.eval(metsig::caloHF,metsig::ET,calotower->et(),calotower->phi(),calotower->eta());
		  sign_tower_sigma_phi = resolutions.eval(metsig::caloHF,metsig::PHI,calotower->et(),calotower->phi(),calotower->eta());
		}
		else{
		  edm::LogWarning("SignCaloSpecificAlgo") << " HCAL tower cell not assigned to an HCAL subdetector!!!" << std::endl;
		}
		// and book!
		metsig::SigInputObj temp(sign_tower_type,sign_tower_et,sign_tower_phi,sign_tower_sigma_et,sign_tower_sigma_phi);
		if(!noHF || subdet !=HcalForward)
		  signInputVec.push_back(temp);
		
		wasused=1;
		hadIsDone = true;
	      }
	    else if( !emIsDone && id.det() == DetId::Ecal )
	      {
		EcalSubdetector subdet = EcalSubdetector( id.subdetId() );
		
		if(subdet == EcalBarrel){
		  sign_tower_type = "emcalotower";
		  sign_tower_et = calotower->emEt();
		  sign_tower_sigma_et = resolutions.eval(metsig::caloEB,metsig::ET,calotower->emEt(),calotower->phi(),calotower->eta());
		  sign_tower_sigma_phi = resolutions.eval(metsig::caloEB,metsig::PHI,calotower->emEt(),calotower->phi(),calotower->eta());
		}
		else if(subdet == EcalEndcap ){
		  sign_tower_type = "emcalotower";
		  sign_tower_et = calotower->emEt();
		  sign_tower_sigma_et = resolutions.eval(metsig::caloEE,metsig::ET,calotower->emEt(),calotower->phi(),calotower->eta());
		  sign_tower_sigma_phi = resolutions.eval(metsig::caloEE,metsig::PHI,calotower->emEt(),calotower->phi(),calotower->eta());
		    
		}
		else{
		  edm::LogWarning("SignCaloSpecificAlgo") << " ECAL tower cell not assigned to an ECAL subdetector!!!" << std::endl;
		}
		metsig::SigInputObj temp(sign_tower_type,sign_tower_et,sign_tower_phi,sign_tower_sigma_et,sign_tower_sigma_phi);
		signInputVec.push_back(temp);
		wasused=1;
		emIsDone = true;
	      }
	  }
	if(wasused==0)
	  edm::LogWarning("SignCaloSpecificAlgo") << "found non-assigned cell, " << std::endl;
      }
    }
  }

  // now run the significance algorithm.
  
  double sign_calo_met_total=0;
  double sign_calo_met_phi=0;
  double sign_calo_met_set=0;
  double significance = metsig::ASignificance(signInputVec, sign_calo_met_total, sign_calo_met_phi, sign_calo_met_set);
  // cleanup everything:
  signInputVec.clear();
  // and return
  return significance;
}
//-------------------------------------------------------------------------

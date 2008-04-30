#include "DataFormats/EgammaCandidates/interface/PhotonID.h"

using namespace reco;


PhotonID::PhotonID(){
  isLooseEM_=false;
  isLoosePhoton_=false;
  isTightPhoton_=false;
  isolationEcalRecHit_=999;
  isolationHcalRecHit_=999;
  isolationSolidTrkCone_=999;
  isolationHollowTrkCone_=999;
  nTrkSolidCone_=999;
  nTrkHollowTrkCone_=999;
  r9_ = 0;
  isEBPho_=false;
  isEEPho_=false;
  isEBGap_=false;
  isEEGap_=false;
  isEBEEGap_=false;
  isAlsoElectron_=false;
}

PhotonID::PhotonID(bool isLooseEM,
		   bool isLoosePho,
		   bool isTightPho, 
		   float TrkCone,
		   float HollowCone, 
		   int nTrkCone, 
		   int nHollow,
		   float EcalRecHitIso,
		   float HcalRecHitIso,
		   float r9,
		   bool EBPho, 
		   bool EEPho, 
		   bool EBGap, 
		   bool EEGap, 
		   bool EBEEGap,
		   bool isAlsoElectron){

  isLooseEM_ = isLooseEM;
  isLoosePhoton_ = isLoosePho;
  isTightPhoton_ = isTightPho;
  isolationEcalRecHit_ = EcalRecHitIso;
  isolationHcalRecHit_ = HcalRecHitIso;
  isolationSolidTrkCone_=TrkCone;
  isolationHollowTrkCone_=HollowCone;
  nTrkSolidCone_=nTrkCone;
  nTrkHollowTrkCone_=nHollow;
  r9_ = r9;
  isEBPho_=EBPho;
  isEEPho_=EEPho;
  isEBGap_=EBGap;
  isEEGap_=EEGap;
  isEBEEGap_=EBEEGap;
  isAlsoElectron_ = isAlsoElectron;
}


void PhotonID::setFiducialFlags(bool EBPho, 
				bool EEPho, 
				bool EBGap, 
				bool EEGap, 
				bool EBEEGap){
  isEBPho_=EBPho;
  isEEPho_=EEPho;
  isEBGap_=EBGap;
  isEEGap_=EEGap;
  isEBEEGap_=EBEEGap;
}


void PhotonID::setDecision(bool decisionLooseEM, bool decisionLoosePho,
			   bool decisionTightPho){
  isLooseEM_ = decisionLooseEM;
  isLoosePhoton_ = decisionLoosePho;
  isTightPhoton_ = decisionTightPho;

}



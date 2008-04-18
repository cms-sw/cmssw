#include "DataFormats/EgammaCandidates/interface/PhotonID.h"

using namespace reco;


PhotonID::PhotonID(){
  cutBasedDecision_=false;
  isolationECal_=999;
  isolationEcalRecHit_=999;
  isolationHcalRecHit_=999;
  isolationSolidTrkCone_=999;
  isolationHollowTrkCone_=999;
  nTrkSolidCone_=999;
  nTrkHollowTrkCone_=999;
  isEBPho_=false;
  isEEPho_=false;
  isEBGap_=false;
  isEEGap_=false;
  isEBEEGap_=false;
  isAlsoElectron_=false;
}

PhotonID::PhotonID(bool Decision, 
		   double BCIso, 
		   double TrkCone,
		   double HollowCone, 
		   int nTrkCone, 
		   int nHollow,
		   double EcalRecHitIso,
		   double HcalRecHitIso,
		   bool EBPho, 
		   bool EEPho, 
		   bool EBGap, 
		   bool EEGap, 
		   bool EBEEGap,
		   bool isAlsoElectron){
  cutBasedDecision_=Decision;
  isolationECal_=BCIso;
  isolationEcalRecHit_ = EcalRecHitIso;
  isolationHcalRecHit_ = HcalRecHitIso;
  isolationSolidTrkCone_=TrkCone;
  isolationHollowTrkCone_=HollowCone;
  nTrkSolidCone_=nTrkCone;
  nTrkHollowTrkCone_=nHollow;
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


void PhotonID::setDecision(bool decision){
  cutBasedDecision_ = decision;
}



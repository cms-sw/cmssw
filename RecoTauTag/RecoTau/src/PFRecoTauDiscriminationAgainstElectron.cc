#include "RecoTauTag/RecoTau/interface/PFRecoTauDiscriminationAgainstElectron.h"

void PFRecoTauDiscriminationAgainstElectron::produce(Event& iEvent,const EventSetup& iEventSetup){
  Handle<PFTauCollection> thePFTauCollection;
  iEvent.getByLabel(PFTauProducer_,thePFTauCollection);

  // fill the AssociationVector object
  auto_ptr<PFTauDiscriminator> thePFTauDiscriminatorAgainstElectron(new PFTauDiscriminator(PFTauRefProd(thePFTauCollection)));

  for(size_t iPFTau=0;iPFTau<thePFTauCollection->size();++iPFTau) {
    PFTauRef thePFTauRef(thePFTauCollection,iPFTau);

    // Check if track goes to Ecal crack
    TrackRef myleadTk;
    if((*thePFTauRef).leadPFChargedHadrCand().isNonnull()){
      myleadTk=(*thePFTauRef).leadPFChargedHadrCand()->trackRef();
      math::XYZPointF myleadTkEcalPos = (*thePFTauRef).leadPFChargedHadrCand()->positionAtECALEntrance();

      if(myleadTk.isNonnull()){ 
	if (applyCut_ecalCrack_ && isInEcalCrack(abs((double)myleadTkEcalPos.eta()))) {
	  thePFTauDiscriminatorAgainstElectron->setValue(iPFTau,0);
	  continue;
	}
      }
    }
    
    bool decision = false;
    bool emfPass = true, htotPass = true, hmaxPass = true; 
    bool h3x3Pass = true, estripPass = true, erecovPass = true;
    bool epreidPass = true, epreid2DPass = true;

    if (applyCut_emFraction_) {
      if ((*thePFTauRef).emFraction() > emFraction_maxValue_) {
	emfPass = false;
      }
    }
    if (applyCut_hcalTotOverPLead_) {
      if ((*thePFTauRef).hcalTotOverPLead() < hcalTotOverPLead_minValue_) {
	htotPass = false;
      }
    }
    if (applyCut_hcalMaxOverPLead_) {
      if ((*thePFTauRef).hcalMaxOverPLead() < hcalMaxOverPLead_minValue_) {
	hmaxPass = false;
      }
    }
    if (applyCut_hcal3x3OverPLead_) {
      if ((*thePFTauRef).hcal3x3OverPLead() < hcal3x3OverPLead_minValue_) {
	h3x3Pass = false;
      }
    }
    if (applyCut_EOverPLead_) {
      if ((*thePFTauRef).ecalStripSumEOverPLead() > EOverPLead_minValue_ &&
	  (*thePFTauRef).ecalStripSumEOverPLead() < EOverPLead_maxValue_) {
	estripPass = false;
      } else {
	estripPass = true;
      }
    }
    if (applyCut_bremsRecoveryEOverPLead_) {
      if ((*thePFTauRef).bremsRecoveryEOverPLead() > bremsRecoveryEOverPLead_minValue_ &&
	  (*thePFTauRef).bremsRecoveryEOverPLead() < bremsRecoveryEOverPLead_maxValue_) {
	erecovPass = false;
      } else {
	erecovPass = true;
      } 
    }
    if (applyCut_electronPreID_) {
      if ((*thePFTauRef).electronPreIDDecision()) {
	epreidPass = false;
      }  else {
	epreidPass = true;
      }
    }
      
    if (applyCut_electronPreID_2D_) {
      if (
	  ((*thePFTauRef).electronPreIDDecision() &&
	   ((*thePFTauRef).ecalStripSumEOverPLead() < elecPreID1_EOverPLead_maxValue ||
	    (*thePFTauRef).hcalTotOverPLead() > elecPreID1_HOverPLead_minValue))
	  ||
	  (!(*thePFTauRef).electronPreIDDecision() &&
	   ((*thePFTauRef).ecalStripSumEOverPLead() < elecPreID0_EOverPLead_maxValue ||
	    (*thePFTauRef).hcalTotOverPLead() > elecPreID0_HOverPLead_minValue))
	  ){
	epreid2DPass = true;
      }  else {
	epreid2DPass = false;
      }
    }

    decision = emfPass && htotPass && hmaxPass && 
      h3x3Pass && estripPass && erecovPass && epreidPass && epreid2DPass;
    if (decision) {
      thePFTauDiscriminatorAgainstElectron->setValue(iPFTau,1);
    } else {
      thePFTauDiscriminatorAgainstElectron->setValue(iPFTau,0);
    }
  }


  iEvent.put(thePFTauDiscriminatorAgainstElectron);
}

bool
PFRecoTauDiscriminationAgainstElectron::isInEcalCrack(double eta) const{  
  return (eta < 0.018 || 
	  (eta>0.423 && eta<0.461) ||
	  (eta>0.770 && eta<0.806) ||
	  (eta>1.127 && eta<1.163) ||
	  (eta>1.460 && eta<1.558));
}

#include "RecoTauTag/RecoTau/interface/PFRecoTauDiscriminationAgainstElectron.h"

void PFRecoTauDiscriminationAgainstElectron::produce(Event& iEvent,const EventSetup& iEventSetup){
  Handle<PFTauCollection> thePFTauCollection;
  iEvent.getByLabel(PFTauProducer_,thePFTauCollection);

  // fill the AssociationVector object
  auto_ptr<PFTauDiscriminator> thePFTauDiscriminatorAgainstElectron(new PFTauDiscriminator(PFTauRefProd(thePFTauCollection)));

  for(size_t iPFTau=0;iPFTau<thePFTauCollection->size();++iPFTau) {
    PFTauRef thePFTauRef(thePFTauCollection,iPFTau);

    bool decision = false;
    bool emfPass = false, htotPass = false, hmaxPass = false, 
      h3x3Pass = false, estripPass = false, erecovPass = false, epreidPass = false;

    if (applyCut_emFraction_) {
      if ((*thePFTauRef).emFraction() < emFraction_maxValue_) {
	emfPass = true;
      }
    }
    if (applyCut_hcalTotOverPLead_) {
      if ((*thePFTauRef).hcalTotOverPLead() > hcalTotOverPLead_minValue_) {
	htotPass = true;
      }
    }
    if (applyCut_hcalMaxOverPLead_) {
      if ((*thePFTauRef).hcalMaxOverPLead() > hcalMaxOverPLead_minValue_) {
	hmaxPass = true;
      }
    }
    if (applyCut_hcal3x3OverPLead_) {
      if ((*thePFTauRef).hcal3x3OverPLead() > hcal3x3OverPLead_minValue_) {
	h3x3Pass = true;
      }
    }
    if (applyCut_ecalStripSumEOverPLead_) {
      if ((*thePFTauRef).ecalStripSumEOverPLead() > ecalStripSumEOverPLead_minValue_ &&
	  (*thePFTauRef).ecalStripSumEOverPLead() < ecalStripSumEOverPLead_maxValue_) {
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
	epreidPass = true;
      }
    }

    decision = emfPass && htotPass && hmaxPass && 
      h3x3Pass && estripPass && erecovPass && epreidPass;
    if (decision) {
      thePFTauDiscriminatorAgainstElectron->setValue(iPFTau,1);
    } else {
      thePFTauDiscriminatorAgainstElectron->setValue(iPFTau,0);
    }
  }


  iEvent.put(thePFTauDiscriminatorAgainstElectron);
}


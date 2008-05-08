#include "RecoTauTag/RecoTau/interface/PFRecoTauDiscriminationAgainstMuon.h"

void PFRecoTauDiscriminationAgainstMuon::produce(Event& iEvent,const EventSetup& iEventSetup){
  Handle<PFTauCollection> thePFTauCollection;
  iEvent.getByLabel(PFTauProducer_,thePFTauCollection);

  // fill the AssociationVector object
  auto_ptr<PFTauDiscriminator> thePFTauDiscriminatorAgainstMuon(new PFTauDiscriminator(PFTauRefProd(thePFTauCollection)));

  for(size_t iPFTau=0;iPFTau<thePFTauCollection->size();++iPFTau) {
    PFTauRef thePFTauRef(thePFTauCollection,iPFTau);
    bool decision = true;

if((*thePFTauRef).hasMuonReference() ){

MuonRef muonref = (*thePFTauRef).leadPFChargedHadrCand()->muonRef();
    if (discriminatorOption_ == "noSegMatch") {
      if ( muonref ->numberOfMatches() > 0 ) {
	decision = false;
      }
    }
    else if (discriminatorOption_ == "twoDCut") {
double seg = muonid::getSegmentCompatibility(*muonref);
double calo= muonref->caloCompatibility(); 
double border = calo * a + seg * b;
      if ( border > 0 ) {
    decision = false; 
      } 
    }
    else if (discriminatorOption_ == "merePresence") decision = false;
    else if (discriminatorOption_ == "combined") { // testing purpose only
unsigned int muType = 0;
if(muonref->isGlobalMuon()) muType = 1;
else if(muonref->isCaloMuon()) muType = 2;
else if(muonref->isTrackerMuon()) muType = 3;
double muonEnergyFraction = (*thePFTauRef).pfTauTagInfoRef()->pfjetRef()->chargedMuEnergyFraction();
bool eta_veto = false;
bool phi_veto = false;
if(fabs(muonref->eta()) > 2.3 || (fabs(muonref->eta()) > 1.4 && fabs(muonref->eta()) < 1.6)) eta_veto = true;
if(muonref->phi() < 0.1 && muonref->phi() > -0.1) phi_veto = true;
if( muType != 1 || muonref ->numberOfMatches() > 0 || eta_veto || phi_veto || muonEnergyFraction > 0.9 ) decision = false; // as place holder
}
else{
//cout << discriminatorOption_ << endl;
throw edm::Exception(edm::errors::UnimplementedFeature) << " Invalid Discriminator Option! Please check cfi file \n";
}

    if (decision) {
      thePFTauDiscriminatorAgainstMuon->setValue(iPFTau,1);
    } else {
      thePFTauDiscriminatorAgainstMuon->setValue(iPFTau,0);
    }
} // valid muon ref
} // pf tau loop
  iEvent.put(thePFTauDiscriminatorAgainstMuon);
} // class itself


#include "JetMETCorrections/TauJet/interface/HardTauCorrector.h"

HardTauCorrector::HardTauCorrector(){
	init();
}
HardTauCorrector::HardTauCorrector(const edm::ParameterSet& iConfig){
	init();
	hardTauAlgorithm->inputConfig(iConfig);
}
HardTauCorrector::~HardTauCorrector(){
	delete hardTauAlgorithm;
}

void HardTauCorrector::init(){
	hardTauAlgorithm = new HardTauAlgorithm;
}

void HardTauCorrector::eventSetup(const edm::Event& iEvent, const edm::EventSetup& iSetup) const {
	hardTauAlgorithm->eventSetup(iEvent,iSetup);
}

double HardTauCorrector::correction(const reco::Jet& fJet, const edm::Event& iEvent, const edm::EventSetup& iSetup) const {

	eventSetup(iEvent,iSetup);
	return correction(fJet);
}

double HardTauCorrector::correction(const math::XYZTLorentzVector& fJet) const{
	return 0;
}
double HardTauCorrector::correction(const reco::Jet& fJet) const {

	if(fJet.et() == 0) return 0;

        TLorentzVector corrected = hardTauAlgorithm->recalculateEnergy(fJet);
        return corrected.Et()/fJet.et();
}

double HardTauCorrector::correction(const reco::CaloJet& fJet) const {

        if(fJet.et() == 0) return 0;

        TLorentzVector corrected = hardTauAlgorithm->recalculateEnergy(fJet);
        return corrected.Et()/fJet.et();
}

double HardTauCorrector::correction(const reco::CaloTau& fJet) const {

        if(fJet.et() == 0) return 0;

        TLorentzVector corrected = hardTauAlgorithm->recalculateEnergy(fJet);
        return corrected.Et()/fJet.et();
}

bool HardTauCorrector::eventRequired() const {
	return true;
}

double HardTauCorrector::efficiency(){
	return hardTauAlgorithm->efficiency();
}

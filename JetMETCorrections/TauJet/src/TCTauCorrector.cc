#include "JetMETCorrections/TauJet/interface/TCTauCorrector.h"

TCTauCorrector::TCTauCorrector(){
	init();
}
TCTauCorrector::TCTauCorrector(const edm::ParameterSet& iConfig, edm::ConsumesCollector &&iC){
	init();
	inputConfig(iConfig, iC);
}
TCTauCorrector::~TCTauCorrector(){
}

void TCTauCorrector::init(){
}

void TCTauCorrector::inputConfig(const edm::ParameterSet& iConfig, edm::ConsumesCollector &iC) {
        tcTauAlgorithm.inputConfig(iConfig, iC);
}

void TCTauCorrector::eventSetup(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
	tcTauAlgorithm.eventSetup(iEvent,iSetup);
}

math::XYZTLorentzVector TCTauCorrector::correctedP4(const reco::CaloTau& fJet) const {
	return tcTauAlgorithm.recalculateEnergy(fJet);
}


/*
double TCTauCorrector::correction(const reco::Jet& fJet, const edm::Event& iEvent, const edm::EventSetup& iSetup) const {

	eventSetup(iEvent,iSetup);
	return correction(fJet);
}
*/

double TCTauCorrector::correction(const math::XYZTLorentzVector& fJet) const{
	return 0;
}

/*
double TCTauCorrector::correction(const reco::Jet& fJet) const {

	if(fJet.et() == 0) return 0;

        math::XYZTLorentzVector corrected = tcTauAlgorithm.recalculateEnergy(fJet);
        return corrected.Et()/fJet.et();
}
*/
/*
double TCTauCorrector::correction(const reco::CaloJet& fJet) const {

        if(fJet.et() == 0) return 0;

        math::XYZTLorentzVector corrected = tcTauAlgorithm.recalculateEnergy(fJet);
        return corrected.Et()/fJet.et();
}
*/


double TCTauCorrector::correction(const reco::CaloTau& fJet) const {

        if(fJet.et() == 0) return 0;

        math::XYZTLorentzVector corrected = tcTauAlgorithm.recalculateEnergy(fJet);
        return corrected.Et()/fJet.et();
}

bool TCTauCorrector::eventRequired() const {
	return true;
}

double TCTauCorrector::efficiency() const {
	return tcTauAlgorithm.efficiency();
}

int TCTauCorrector::allTauCandidates() const {
        return tcTauAlgorithm.allTauCandidates();
}

int TCTauCorrector::statistics() const{
        return tcTauAlgorithm.statistics();
}


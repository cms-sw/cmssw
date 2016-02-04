#include "JetMETCorrections/TauJet/interface/TCTauCorrector.h"

TCTauCorrector::TCTauCorrector(){
	init();
}
TCTauCorrector::TCTauCorrector(const edm::ParameterSet& iConfig){
	init();
	inputConfig(iConfig);
}
TCTauCorrector::~TCTauCorrector(){
	delete tcTauAlgorithm;
}

void TCTauCorrector::init(){
	tcTauAlgorithm = new TCTauAlgorithm;
}

void TCTauCorrector::inputConfig(const edm::ParameterSet& iConfig) const {
        tcTauAlgorithm->inputConfig(iConfig);
}

void TCTauCorrector::eventSetup(const edm::Event& iEvent, const edm::EventSetup& iSetup) const {
	tcTauAlgorithm->eventSetup(iEvent,iSetup);
}

math::XYZTLorentzVector TCTauCorrector::correctedP4(const reco::CaloTau& fJet) const {
	return tcTauAlgorithm->recalculateEnergy(fJet);
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

        math::XYZTLorentzVector corrected = tcTauAlgorithm->recalculateEnergy(fJet);
        return corrected.Et()/fJet.et();
}
*/
/*
double TCTauCorrector::correction(const reco::CaloJet& fJet) const {

        if(fJet.et() == 0) return 0;

        math::XYZTLorentzVector corrected = tcTauAlgorithm->recalculateEnergy(fJet);
        return corrected.Et()/fJet.et();
}
*/
double TCTauCorrector::correction(const reco::CaloTau& fJet) const {

        if(fJet.et() == 0) return 0;

        math::XYZTLorentzVector corrected = tcTauAlgorithm->recalculateEnergy(fJet);
        return corrected.Et()/fJet.et();
}

bool TCTauCorrector::eventRequired() const {
	return true;
}

double TCTauCorrector::efficiency(){
	return tcTauAlgorithm->efficiency();
}

int TCTauCorrector::allTauCandidates(){
        return tcTauAlgorithm->allTauCandidates();
}

int TCTauCorrector::statistics(){
        return tcTauAlgorithm->statistics();
}

int TCTauCorrector::algoComponent(){
        return tcTauAlgorithm->algoComponent();
}

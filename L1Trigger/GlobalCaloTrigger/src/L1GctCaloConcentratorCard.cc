#include "../interface/L1GctCaloConcentratorCard.h"

L1GctCaloConcentratorCard::L1GctCaloConcentratorCard() {
	isoElectronLeafCard = new L1GctElectronLeafCard();
	nonIsoElectronLeafCard = new L1GctElectronLeafCard();
}

L1GctCaloConcentratorCard::~L1GctCaloConcentratorCard() {
	delete isoElectronLeafCard;
	delete nonIsoElectronLeafCard;	
}

vector<L1GctEmCand> L1GctCaloConcentratorCard::getIsoElectrons() {
	return isoElectronLeafCard->getOutput();
}

vector<L1GctEmCand> L1GctCaloConcentratorCard::getNonIsoElectrons() {
	return isoElectronLeafCard->getOutput();
}

vector<L1GctJet> L1GctCaloConcentratorCard::getCentralJets() {
	return jetFinalStage.getCentralJets();
}

vector<L1GctJet> L1GctCaloConcentratorCard::getForwardJets() {
	return jetFinalStage.getForwardJets();	
}

vector<L1GctJet> L1GctCaloConcentratorCard::getTauJets() {
	return jetFinalStage.getTauJets();	
}
	
unsigned L1GctCaloConcentratorCard::getEtMiss() {
	return globalEnergyAlgos.getEtMiss();
}

unsigned L1GctCaloConcentratorCard::getEtMissPhi() {
	return globalEnergyAlgos.getEtMissPhi();	
}

unsigned L1GctCaloConcentratorCard::getEtSum() {
	return globalEnergyAlgos.getEtSum();	
}

unsigned L1GctCaloConcentratorCard::getEtHad() {
	return globalEnergyAlgos.getEtHad();	
}


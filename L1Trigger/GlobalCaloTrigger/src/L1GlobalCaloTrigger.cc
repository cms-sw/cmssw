#include "L1Trigger/GlobalCaloTrigger/interface/L1GlobalCaloTrigger.h"

L1GlobalCaloTrigger* L1GlobalCaloTrigger::instance = 0;

// constructor
L1GlobalCaloTrigger::L1GlobalCaloTrigger() :
	theJetFinders(18),
	theElectronSorters(4),
	theWheelJetFpgas(2),
	theWheelEnergyFpgas(2),
{
	
	// for first release, just instantiate algorithms
	for (int i=0; i<18; i++) {
		theJetFinders[i] = new L1GctJetFinder();
	}

	for (int i=0; i<4; i++) {
		theElectronSorters[i] = new L1GctElectronSorter();
	}		
	
	for (int i=0; i<2; i++) {
		theWheelJetFpgas[i] = new L1GctWheelJetFpga();
		theWheelEnergyFpgas[i] = new L1GctWheelEnergyFpga();
	}
	
	theJetFinalStage = new L1GctJetFinalStage();
	theElectronFinalSort = new L1GctElectronFinalSort();
	theGlobalEnergyAlgos = new L1GctGlobalEnergyAlgos();
	
	// instantiate Source Cards	
//	for (int i=0; i<18; i++) {
//		L1GctSourceCard* sc = new L1GctSourceCard();
//		theSourceCards.push_back(sc);
//	}
	
}

L1GlobalCaloTrigger::~L1GlobalCaloTrigger()
{
	delete caloConcCard;
	delete muonConcCard;

	delete plusWheelCard;
	delete minusWheelCard;
	
	theSourceCards.clear();
}

L1GlobalCaloTrigger* L1GlobalCaloTrigger::theGct() {
	if (L1GlobalCaloTrigger::instance==0) {
		L1GlobalCaloTrigger::instance = new L1GlobalCaloTrigger();
	}
	return L1GlobalCaloTrigger::instance;
}

void L1GlobalCaloTrigger::process() {
		
//	// Leaf Card processing
//	for (int i=0; i<18; i++) {
//			theJetFinders[i]->process();
//	}
//	
//	for (int i=0; i<4; i++) {
//			theElectronSorters[i]->process();
//	}	
//	
//	// Wheel Card processing
//	for (int i=0; i<2; i++) {  // Wheel cards
//		for (int j=0; j<9; j++) { // jet finders 
//			for (int k=0; k<4; k++) {  // jets
//				theWheelJetFpgas[i]->setInputJet(4*j+k, theJetFinders[9*i+j]->getOutput()[k]);
//			}
//		}
//		theWheelJetFpgas[i]->process();
//	}
//
//	for (int i=0; i<2; i++) { // Wheel Cards
//		for (int j=0; j<3; j++) {	 // Leaf Cards
//			for (int k=0; k<3; k++) { // Jet Finders
//				theWheelEnergyFpgas[i]->setInputEx(k, theJetFinders[9*i+3*j+k]->getOutputEx());
//				theWheelEnergyFpgas[i]->setInputEy(k, theJetFinders[9*i+3*j+k]->getOutputEy());
//				theWheelEnergyFpgas[i]->setInputEt(k, theJetFinders[9*i+3*j+k]->getOutputEt());
//			}
//		}
//		theWheelEnergyFpgas[i]->process();
//	}
//	
//	// Concentrator Card processing	
//	for (int i=0; i<4; i++) {
//		for (int j=0; j<4; j++) {
//			theElectronFinalSort.setInput(4*i+j, theElectronSorters[i]->getOutput()[j];
//		}
//	}
//	theElectronFinalSort->process();
//		
//	for (int i=0; i<2; i++) {	
//		theGlobalEnergyAlgos->setInputEx(i, theWheelEnergyFpgas[i]->getOutputEx());
//		theGlobalEnergyAlgos->setInputEy(i, theWheelEnergyFpgas[i]->getOutputEy());
//		theGlobalEnergyAlgos->setInputEt(i, theWheelEnergyFpgas[i]->getOutputEt());
//		theGlobalEnergyAlgos->setInputHt(i, theWheelEnergyFpgas[i]->getOutputHt());
//	}
//	theGlobalEnergyAlgos->process();
	
	
}

#ifndef L1GLOBALCALOTRIGGER_H_
#define L1GLOBALCALOTRIGGER_H_

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctCaloConcentratorCard.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctElectronFinalSort.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctMuonConcentratorCard.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctWheelCard.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctWheelJetFpga.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctWheelEnergyFpga.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctElectronLeafCard.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctMuonLeafCard.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctSourceCard.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctEmCand.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJet.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctEvent.h"

#include <vector>

using namespace std;

/**
  * Represents the GCT system
  * This is the main point of access for the user
  * 
  * author: Jim Brooke
  * date: 20/2/2006
  * 
  **/ 

class L1GlobalCaloTrigger {
public:

	~L1GlobalCaloTrigger();

	static L1GlobalCaloTrigger* theGct();
	
	void process();
	
	// outputs to Global Trigger
	vector<L1GctEmCand> getIsoElectrons();
	vector<L1GctEmCand> getNonIsoElectrons();
	vector<L1GctJet> getCentralJets();
	vector<L1GctJet> getForwardJets();
	vector<L1GctJet> getTauJets();

	unsigned getEtMiss();
	unsigned getEtMissPhi();
	unsigned getEtSum();
	unsigned getEtHad();
	
	// DAQ output - what owns the produced event?
	L1GctEvent getEvent();

	// access the hardware
	inline L1GctCaloConcentratorCard* getCaloConcCard() { return caloConcCard; }
	inline L1GctMuonConcentratorCard* getMuonConcCard() { return muonConcCard; }
//	inline L1GctElectronLeafCard* getIsoElectronLeafCard() { return caloConcCard->getIsoElectronLeafCard; }
//	inline L1GctElectronLeafCard* getNonIsoElectronLeafCard() { return caloConcCard->getNonIsoElectronLeafCard; }
	inline L1GctWheelCard* getPlusWheelCard() { return plusWheelCard; }
	inline L1GctWheelCard* getMinusWheelCard() { return minusWheelCard; }
//	inline vector<L1GctJetLeafCard*> getPlusJetLeafCards { return plusWheelCard->getJetLeafCards; }
//	inline vector<L1GctJetLeafCard*> getMinusJetLeafCards { return minusWheelCard->getJetLeafCards; }
	inline vector<L1GctSourceCard*> getSourceCards() { return theSourceCards; }

private:

	// singleton private constructor
	L1GlobalCaloTrigger();
	
	// move data around
	void setupAlgoInputs();

private:

	// instance of the GCT
	static L1GlobalCaloTrigger* instance;

	// pointers to the algorithms
	vector<L1GctJetFinder*> theJetFinders;
	vector<L1GctElectronSorter*> theElectronSorters;
	vector<L1GctWheelJetFpga*> theWheelJetFpgas;
	vector<L1GctWheelEnergyFpga*> theWheelEnergyFpgas;
	L1GctJetFinalStage* theJetFinalStage;
	L1GctGlobalEnergyAlgos* theGlobalEnergyAlgos;
	L1GctElectronFinalSort* theElectronFinalSort;


	// pointers to the modules
	L1GctCaloConcentratorCard* caloConcCard;
	L1GctMuonConcentratorCard* muonConcCard;
	
	L1GctWheelCard* plusWheelCard;
	L1GctWheelCard* minusWheelCard;	
	
	vector<L1GctSourceCard*> theSourceCards;
	
};

#endif /*L1GLOBALCALOTRIGGER_H_*/

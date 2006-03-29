#ifndef L1GLOBALCALOTRIGGER_H_
#define L1GLOBALCALOTRIGGER_H_

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctEmCand.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJet.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctEvent.h"

#include <vector>

using std::vector;

/**
  * Represents the GCT system
  * This is the main point of access for the user
  * 
  * author: Jim Brooke
  * date: 20/2/2006
  * 
  **/ 

class L1GctSourceCard;

class L1GctElectronSorter;
class L1GctElectronFinalSort;

class L1GctJetFinder;
class L1GctWheelJetFpga;
class L1GctJetFinalStage;

class L1GctWheelEnergyFpga;
class L1GctGlobalEnergyAlgos;

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

	//vector<L1GctJetLeafCard*> getPlusJetLeafCards();
	//vector<L1GctJetLeafCard*> getMinusJetLeafCards();

	vector<L1GctSourceCard*> getSourceCards();

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
	vector<L1GctSourceCard*> theSourceCards;
	
};

#endif /*L1GLOBALCALOTRIGGER_H_*/

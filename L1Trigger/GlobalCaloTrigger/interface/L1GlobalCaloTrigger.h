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

class L1GctJetLeafCard;
class L1GctWheelJetFpga;
class L1GctJetFinalStage;

class L1GctWheelEnergyFpga;
class L1GctGlobalEnergyAlgos;

class L1GlobalCaloTrigger {
public:
	///
	/// destruct the GCT
	~L1GlobalCaloTrigger();
	///
	/// get the GCT object
	static L1GlobalCaloTrigger* theGct();
	///
	/// process an event
	void process();
	///
	/// iso electron outputs to GT
	vector<L1GctEmCand> getIsoElectrons();
	/// 
	/// non-iso electron outputs to GT
	vector<L1GctEmCand> getNonIsoElectrons();
	///
	/// central jet outputs to GT
	vector<L1GctJet> getCentralJets();
	///
	/// forward jet outputs to GT
	vector<L1GctJet> getForwardJets();
	///
	/// tau jet outputs to GT
	vector<L1GctJet> getTauJets();
	///
	/// Etmiss output to GT
	unsigned getEtMiss();
	///
	/// Etmiss phi output to GT
	unsigned getEtMissPhi();
	///
	/// Total Et output to GT
	unsigned getEtSum();
	///
	/// Total hadronic Et output to GT
	unsigned getEtHad();
	///
	/// DAQ output - what owns the produced event?
	L1GctEvent getEvent();
	///
	/// get the Source cards
	vector<L1GctSourceCard*> getSourceCards() { return theSourceCards; }
	///
	/// get the Jet Leaf cards
	vector<L1GctJetLeafCard*> getJetLeafCards() { return theJetLeafCards; }

private:
	///
	/// singleton private constructor
	L1GlobalCaloTrigger();
	///
	/// instantiate the hardware & algo objects
	void build();
	///
	/// wire up the hardware obejcts
	void setup();

private:

	// instance of the GCT
	static L1GlobalCaloTrigger* instance;

	// pointers to the hardware/algos
	vector<L1GctSourceCard*> theSourceCards;
	vector<L1GctJetLeafCard*> theJetLeafCards;			
	vector<L1GctElectronSorter*> theElectronSorters;	
	vector<L1GctWheelJetFpga*> theWheelJetFpgas;		
	vector<L1GctWheelEnergyFpga*> theWheelEnergyFpgas;

	///
	/// central barrel jet find & final sort
	L1GctJetFinalStage* theJetFinalStage;			
	///
	/// energy final stage algos
	L1GctGlobalEnergyAlgos* theEnergyFinalStage;	
	///
	/// electron final stage sorters (0 = iso, 1 = non-iso)
	vector<L1GctElectronFinalSort*> theElectronFinalStage;
	
};

#endif /*L1GLOBALCALOTRIGGER_H_*/

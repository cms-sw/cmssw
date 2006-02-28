#ifndef L1GCTCALOCONCENTRATORCARD_H_
#define L1GCTCALOCONCENTRATORCARD_H_

#include "L1GctElectronLeafCard.h"
#include "L1GctWheelCard.h"

#include "L1GctJetFinalStage.h"
#include "L1GctGlobalEnergyAlgos.h"

#include "L1GctEmCand.h"
#include "L1GctJet.h"

#include <vector>

using namespace std;

/*
 * Represents a GCT Concentrator Card
 * programmed to process Calo data
 * author: Jim Brooke
 * date: 20/2/2006
 * 
 */

class L1GctCaloConcentratorCard
{
public:

	L1GctCaloConcentratorCard();
	virtual ~L1GctCaloConcentratorCard();

	void process();
	
	vector<L1GctEmCand> getIsoElectrons();
	vector<L1GctEmCand> getNonIsoElectrons();	
	vector<L1GctJet> getCentralJets();
	vector<L1GctJet> getForwardJets();
	vector<L1GctJet> getTauJets();
	
	unsigned getEtMiss();
	unsigned getEtMissPhi();
	unsigned getEtSum();
	unsigned getEtHad();
	
private:

	// internal processing
	L1GctJetFinalStage jetFinalStage;
	L1GctGlobalEnergyAlgos globalEnergyAlgos;

	// pointers to upstream data sources
	L1GctElectronLeafCard* isoElectronLeafCard;
	L1GctElectronLeafCard* nonIsoElectronLeafCard;
	
	L1GctWheelCard* plusWheelCard;
	L1GctWheelCard* minusWheelCard;
	
	
};

#endif /*L1GCTCALOCONCENTRATORCARD_H_*/

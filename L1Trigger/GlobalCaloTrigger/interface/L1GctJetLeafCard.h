#ifndef L1GCTJETLEAFCARD_H_
#define L1GCTJETLEAFCARD_H_

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetCand.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetFinder.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctSourceCard.h"

#include <vector>
#include <bitset>

using std::vector;
using std::bitset;

/*
 * Represents a GCT Leaf Card
 * programmed for jet finding
 * author: Jim Brooke
 * date: 20/2/2006
 * 
 */

class L1GctJetLeafCard : L1GctProcessor
{
public:
	L1GctJetLeafCard(int id, int iphi);
	~L1GctJetLeafCard();

	/// clear internal buffers
	virtual void reset();

	/// set the input buffers
	virtual void fetchInput();
 
	/// process the data and set outputs
	virtual void process();

	/// add a Source Card
	void setInputSourceCard(int i, L1GctSourceCard* card);

	/// get the input data
	vector<L1GctRegion> getInputRegions();
    
	// get the jet output
	vector<L1GctJetCand> getOutputJetsA() { jetFinderA->getJets(); }  ///< Output jetfinder A jets (lowest jetFinder in phi)
	vector<L1GctJetCand> getOutputJetsB() { jetFinderB->getJets(); }  ///< Output jetfinder B jets (middle jetFinder in phi)
	vector<L1GctJetCand> getOutputJetsC() { jetFinderC->getJets(); }  ///< Ouptut jetfinder C jets (highest jetFinder in phi)
    
	/// get the Ex output
	inline unsigned long getOutputEx() { return outputEx.to_ulong(); }
	
	/// get the Ey output
	inline unsigned long getOutputEy() { return outputEy.to_ulong(); }
	
	/// get the Et output
	inline unsigned long getOutputEt() { return outputEt.to_ulong(); }
	inline unsigned long getOutputHt() { return outputHt.to_ulong(); }
    
    static const int MAX_JET_FINDERS = 3;

private:

	// Leaf card ID
	int m_id;

	// internal algorithms
	L1GctJetFinder* jetFinderA;  ///< lowest jetFinder in phi
	L1GctJetFinder* jetFinderB;  ///< middle jetFinder in phi
	L1GctJetFinder* jetFinderC;  ///< highest jetFinder in phi

	// pointers to data source
	vector<L1GctSourceCard*> m_sourceCards;

	// internal data (other than jets)
	static const int NUM_BITS_ENERGY_DATA = 13;
	static const int OVERFLOW_BIT = NUM_BITS_ENERGY_DATA - 1;

	static const int Emax = (1<<NUM_BITS_ENERGY_DATA);
	static const int signedEmax = (Emax>>1);

	int phiPosition;

	bitset<NUM_BITS_ENERGY_DATA> outputEx;
	bitset<NUM_BITS_ENERGY_DATA> outputEy;
	bitset<NUM_BITS_ENERGY_DATA> outputEt;
	bitset<NUM_BITS_ENERGY_DATA> outputHt;
};

#endif /*L1GCTJETLEAFCARD_H_*/

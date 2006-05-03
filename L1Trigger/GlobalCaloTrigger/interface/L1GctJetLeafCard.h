#ifndef L1GCTJETLEAFCARD_H_
#define L1GCTJETLEAFCARD_H_

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetFinder.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctSourceCard.h"

#include <vector>

using namespace std;

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
	L1GctJetLeafCard(int iphi);
	~L1GctJetLeafCard();
	///
	/// clear internal buffers
	virtual void reset();
	///
	/// set the input buffers
	virtual void fetchInput();
	/// 
	/// process the data and set outputs
	virtual void process();
	///
	/// add a Source Card
	void setInputSourceCard(int i, L1GctSourceCard* card);
	///
	/// get the input data
	vector<L1GctRegion> getInputRegions();
	///
	/// get the jet output
	vector<L1GctJet> getOutputJets();
	///
	/// get the energy outputs
	inline unsigned long getOutputEx() { return outputEx.to_ulong(); }
	inline unsigned long getOutputEy() { return outputEy.to_ulong(); }
	inline unsigned long getOutputEt() { return outputEt.to_ulong(); }

private:

	// internal algorithms
	L1GctJetFinder* jetFinderA;
	L1GctJetFinder* jetFinderB;
	L1GctJetFinder* jetFinderC;	

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
};

#endif /*L1GCTJETLEAFCARD_H_*/

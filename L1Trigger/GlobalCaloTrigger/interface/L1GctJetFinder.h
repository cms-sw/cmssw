#ifndef L1GCTJETFINDER_H_
#define L1GCTJETFINDER_H_

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJet.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctRegion.h"

#include <vector>

/*! \class testJetFinder.cpp
 * \brief 3*3 sliding window algorithm jet finder.
 *
 *  Locates the jets from 48 inputted L1GctRegions.
 *  This uses the 3*3 sliding window algorithm.
 * 
 *  Inputted regions are expected in a certain order with respect
 *  to the index i:
 * 
 *  Regions should arrive running from the middle (eta=0) of the detector
 *  out towards the edge of the forward HCAL, and then moving across
 *  in columns like this but increasing in phi each time.
 * 
 *  E.g. for 48 inputted regions:
 *       region  0: phi=0, other side of eta=0 line (shared data).
 *       region  1: phi=0, but correct side of eta=0 (shared data).
 *       .
 *       . 
 *       region 11: phi=0, edge of Forward HCAL (shared data).
 *       region 12: phi=20, other side of eta=0 line (shared data)
 *       region 13: phi=20, start of jet search area
 *       .
 *       .
 *       region 23: phi=20, edge of HF (jet search area)
 *       etc.
 * 
 *  In the event of neighbouring regions having the same energy, this
 *  will locate the jet in the region closest to eta=0 that has the
 *  lowest value of phi.
 */
/*
 * \author Jim Brooke & Robert Frazier
 * \date March 2006
 */

//Typedefs
typedef unsigned long int ULong;
typedef unsigned short int UShort;

//Constants
//Controls the maximum number of input regions allowed
const int maxRegionsIn = 48; // 2*11 search area, so 4*12=48 regions needed to run search.

const int columnOffset = maxRegionsIn/4;  //The index offset between columns


class L1GctJetFinder
{
public:
	L1GctJetFinder();
	~L1GctJetFinder();

	/// Clears internal data
	void reset();
    
    /// Set input data
    void setInputRegion(int i, L1GctRegion region);
	
	/// Process the event (run jet algorithm)
	void process();

	// Return input data	
	vector<L1GctRegion> getInputRegions() const { return m_inputRegions; }

	// Return output data
	vector<L1GctJet> getJets() const { return m_outputJets; }
	ULong getHt() const { return m_outputHt.to_ulong(); }
	
	// need method(s) to return jet counts - need to decide type!
		
private:

	// input data (this may need to go on the heap...)
	vector<L1GctRegion> m_inputRegions;

	// output jets
	vector<L1GctJet> m_outputJets;  //6 maximum for 48 regions, but will use pushback

	// output Ht - need to confirm number of bits
	bitset<12> m_outputHt;
	
	// jet count output - need to decide data type!
	//vector<bitset<4>> outputJetCounts;
	
    
    //PRIVATE METHODS
    /// Returns true if region index is the centre of a jet. Set boundary = true if at edge of HCAL.
    bool detectJet(const UShort centreIndex, const bool boundary = false) const;

    /// Returns energy sum (rank) of the 9 regions centred (physically) about centreIndex. Set boundary = true if at edge of HCAL.
    ULong calcJetRank(const UShort centreIndex, const bool boundary = false) const;

    /// Returns combined tauVeto of the 9 regions centred (physically) about centreIndex. Set boundary = true if at edge of HCAL.
    bool calcJetTauVeto(const UShort centreIndex, const bool boundary = false) const;
    
    /// Converts a 10-bit energy to a 6-bit calibrated rank - rather arbitrarily at the mo.
    ULong convertToRank(const ULong energy) const;
    
    /// Calculates total calibrated energy in jets (Ht) sum
    ULong calcHt() const;
    
};

#endif /*L1GCTJETFINDER_H_*/

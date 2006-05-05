#ifndef L1GCTJETFINDER_H_
#define L1GCTJETFINDER_H_

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJet.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctRegion.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctProcessor.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctSourceCard.h"

#include <vector>

/*! \class L1GctJetFinder
 * \brief 3*3 sliding window algorithm jet finder.
 *
 *  Locates the jets from 48 inputted L1GctRegions.
 *  This uses the 3*3 sliding window algorithm.
 * 
 *  SourceCard pointers should be set up according to:
 *  http://frazier.home.cern.ch/frazier/wiki_resources/GCT/jetfinder_sourcecard_wiring.jpg
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
 *  will locate the jet in the region furthest from eta=0 that has the
 *  lowest value of phi.
 */
/*
 * \author Jim Brooke & Robert Frazier
 * \date March 2006
 */



class L1GctJetFinder : public L1GctProcessor
{
public:
    //Typedefs
    typedef unsigned long int ULong;
    typedef unsigned short int UShort;
    
    ///Type for telling the jet finder which half of the detector it is in, eta < 0 or eta > 0.
    enum EtaHalf {NEG_ETA_TYPE=1, POS_ETA_TYPE};

    ///Jetfinder needs to know which half of the detector it is in to properly load data from sourcecards.
    L1GctJetFinder(int id, EtaHalf etaHalf);
    ~L1GctJetFinder();
   
    /// clear internal buffers
    virtual void reset();

    /// get input data from sources
    virtual void fetchInput();

    /// process the data, fill output buffers
    virtual void process();

    /// set an input Source Card pointer 
    void setInputSourceCard(int i, L1GctSourceCard* sc);

    /// Set input data
    void setInputRegion(int i, L1GctRegion region);
    
    /// Return input data   
    std::vector<L1GctRegion> getInputRegions() const { return m_inputRegions; }

    /// Return output data
    std::vector<L1GctJet> getJets() const { return m_outputJets; }
    ULong getHt() const { return m_outputHt.to_ulong(); }
    
    // need method(s) to return jet counts - need to decide type!
        
private:

	///
	/// algo ID
	int m_id;
	
    //Constants
    static const int maxSourceCards = 9;  //need data from 9 separate source cards to find jets in the 2*11 search region.
    static const int maxRegionsIn = 48; // 2*11 search area, so 4*12=48 regions needed to run search.
    static const int columnOffset = maxRegionsIn/4;  //The index offset between columns
    static const int maxJets = 6;  //max of 6 jets in a 2*11 search area

    ///Which half of the detector we are in
    EtaHalf m_etaHalf;

    /// Store source card pointers
    std::vector<L1GctSourceCard*> m_sourceCards;
    
    /// input data required for jet finding
    std::vector<L1GctRegion> m_inputRegions;

    /// output jets
    std::vector<L1GctJet> m_outputJets;

    /// output Ht - need to confirm number of bits
    std::bitset<12> m_outputHt;
    
    // jet count output - need to decide data type!
    //vector<bitset<4>> outputJetCounts;
    
    
    //PRIVATE METHODS
    /// Returns true if region index is the centre of a jet. Set boundary = true if at edge of HCAL.
    bool detectJet(const UShort centreIndex, const bool boundary = false) const;

    /// Returns energy sum (rank) of the 9 regions centred (physically) about centreIndex. Set boundary = true if at edge of HCAL.
    ULong calcJetRank(const UShort centreIndex, const bool boundary = false) const;

    /// Returns combined tauVeto of the 9 regions centred (physically) about centreIndex. Set boundary = true if at edge of Endcap.
    bool calcJetTauVeto(const UShort centreIndex, const bool boundary = false) const;
    
    /// Converts a 10-bit energy to a 6-bit calibrated rank - rather arbitrarily at the mo.
    ULong convertToRank(const ULong energy) const;
    
    /// Calculates total calibrated energy in jets (Ht) sum
    ULong calcHt() const;
    
};

#endif /*L1GCTJETFINDER_H_*/

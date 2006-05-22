#ifndef L1GCTJETFINDER_H_
#define L1GCTJETFINDER_H_

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetCand.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctRegion.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctProcessor.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctSourceCard.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctEtTypes.h"

#include <vector>

using std::binary_function;

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
 * 
 *  The jet finder outputs jets with a local eta/phi co-ordinate system 
 *  for the jets it finds in the 2*11 search area it is looking in.
 *  Eta runs from 0 to 10, with 0 being the region closest to the central
 *  eta=0 line, and 10 being the edge of which ever half of the detector
 *  the jet finder is operating in at eta= +/-5.  Phi data is set to either 
 *  0 or 1, to indicate increasing real-world phi co-ordinate.
 *  
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
    
    /// id is 0-8 for -ve Eta jetfinders, 9-17 for +ve Eta, for increasing Phi.
    L1GctJetFinder(int id);
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
    std::vector<L1GctJetCand> getJets() const { return m_outputJets; }
    L1GctScalarEtVal ht() const { return m_outputHt; }
    
    ///Max. number of jets that can be found per jet finder
    static const int MAX_JETS_OUT = 6;  //max of 6 jets in a 2*11 search area

private:

	/// algo ID
	int m_id;
	
    //Constants
    static const int MAX_SOURCE_CARDS = 9;  //need data from 9 separate source cards to find jets in the 2*11 search region.
    static const int MAX_REGIONS_IN = 48; // 2*11 search area, so 4*12=48 regions needed to run search.
    static const int COL_OFFSET = MAX_REGIONS_IN/4;  ///< The index offset between columns

    /// Store source card pointers
    std::vector<L1GctSourceCard*> m_sourceCards;
    
    /// input data required for jet finding
    std::vector<L1GctRegion> m_inputRegions;

    /// output jets
    std::vector<L1GctJetCand> m_outputJets;

    /// output Ht
    L1GctScalarEtVal m_outputHt;
    

    
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
    L1GctScalarEtVal calcHt() const;
  
};

#endif /*L1GCTJETFINDER_H_*/

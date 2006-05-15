#ifndef L1GCTWHEELJETFPGA_H_
#define L1GCTWHEELJETFPGA_H_

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetCand.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctProcessor.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetLeafCard.h"

#include <vector>
#include <bitset>

class L1GctWheelJetFpga : public L1GctProcessor
{
public:
    L1GctWheelJetFpga(int id);
    ~L1GctWheelJetFpga();

    typedef std::vector<L1GctJetCand> JetVector;

    /// clear internal buffers
    virtual void reset();

    /// get input data from sources
    virtual void fetchInput();

    /// process the data, fill output buffers
    virtual void process();

    /// set input sources
    void setInputLeafCard(int i, L1GctJetLeafCard* card);

    /// set input data      
    void setInputJet(int i, L1GctJetCand jet); 
    void setInputHt (int i, unsigned ht);
    
    /// get the input jets. Jets 0-5 from leaf card 0, jetfinderA.  Jets 6-11 from leaf card 0, jetfinder B... etc.
    JetVector getInputJets() const { return m_inputJets; }
    
    // get the input Ht
    unsigned long getInputHt(unsigned leafnum) const { return m_inputHt[leafnum].to_ulong(); }
    
    // get the output jets
    JetVector getCentralJets() const { return m_centralJets; }
    JetVector getForwardJets() const { return m_forwardJets; }
    JetVector getTauJets() const { return m_tauJets; }
    
    // get the output Ht and jet counts
    unsigned long getOutputHt() const { return m_outputHt.to_ulong(); }
    unsigned long getOutputJc(unsigned jcnum) const { return m_outputJc[jcnum].to_ulong(); }
    
private:
    /// Max number of leaf card pointers
    static const int MAX_LEAF_CARDS = 3;
    /// Maximum number of jets we can have as input
    static const int MAX_JETS_IN = MAX_LEAF_CARDS * L1GctJetLeafCard::MAX_JET_FINDERS * L1GctJetFinder::MAX_JETS_OUT;
    /// Max number of jets of each type (central, foward, tau) we output.
    static const int MAX_JETS_OUT = 4;
    
    /// algo ID
    int m_id;

    /// the jet leaf cards
    std::vector<L1GctJetLeafCard*> m_inputLeafCards;
    
    /// input data. Jets 0-5 from leaf card 0, jetfinderA.  Jets 6-11 from leaf card 0, jetfinder B... etc.
    JetVector m_inputJets;
    
    // Holds the all the various inputted jets, re-addressed using proper GCT->GT jet addressing
    JetVector m_rawCentralJets; 
    JetVector m_rawForwardJets; 
    JetVector m_rawTauJets; 

    // input Ht sums from each leaf card
    static const int NUM_BITS_ENERGY_DATA = 13;
    static const int OVERFLOW_BIT = NUM_BITS_ENERGY_DATA - 1;

    static const int Emax = (1<<NUM_BITS_ENERGY_DATA);
    static const int signedEmax = (Emax>>1);

    // input data - need to confirm number of bits!
    typedef std::bitset<NUM_BITS_ENERGY_DATA> InputEnergyType;
    std::vector<InputEnergyType> m_inputHt;

    // output data
    JetVector m_centralJets;
    JetVector m_forwardJets;
    JetVector m_tauJets;
    
    // data sent to GlobalEnergyAlgos
    typedef std::bitset<3> JcWheelType;
    std::bitset<NUM_BITS_ENERGY_DATA> m_outputHt;
    std::vector<JcWheelType> m_outputJc;
      
    //PRIVATE METHODS
    /// Puts the output from a jetfinder into the correct index range of the m_inputJets array. 
    void storeJets(JetVector jets, unsigned short iLeaf, unsigned short offset);
    /// Classifies jets into central, forward or tau, and re-addresses them using global co-ords.
    void classifyJets();  
};

#endif /*L1GCTWHEELJETFPGA_H_*/

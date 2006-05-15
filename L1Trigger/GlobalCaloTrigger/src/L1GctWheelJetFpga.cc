#include "L1Trigger/GlobalCaloTrigger/interface/L1GctWheelJetFpga.h"

using std::vector;
using std::bitset;

L1GctWheelJetFpga::L1GctWheelJetFpga(int id):
    m_id(id),
    m_inputLeafCards(MAX_LEAF_CARDS),
    m_inputJets(MAX_JETS_IN),
    m_inputHt(MAX_LEAF_CARDS),
    m_centralJets(MAX_JETS_OUT),
    m_forwardJets(MAX_JETS_OUT),
    m_tauJets(MAX_JETS_OUT),
    m_outputJc(12)
{
}

L1GctWheelJetFpga::~L1GctWheelJetFpga()
{
}

void L1GctWheelJetFpga::reset()
{
    m_inputJets.clear();
    m_inputJets.resize(MAX_JETS_IN);
    
    m_rawCentralJets.clear();
    m_rawForwardJets.clear();
    m_rawTauJets.clear();
    
    m_centralJets.clear();
    m_centralJets.resize(MAX_JETS_OUT);
    m_forwardJets.clear();
    m_forwardJets.resize(MAX_JETS_OUT);
    m_tauJets.clear();
    m_tauJets.resize(MAX_JETS_OUT);
    
    for (int i=0; i<MAX_LEAF_CARDS; ++i) 
    {
        m_inputHt[i].reset();
    }
    m_outputHt.reset();
    for (int i=0; i<12; ++i) 
    {
        m_outputJc[i].reset();
    }
}

void L1GctWheelJetFpga::fetchInput()
{
    //Get Jets
    for(unsigned short iLeaf = 0; iLeaf < MAX_LEAF_CARDS; ++iLeaf)
    {
        assert(m_inputLeafCards[iLeaf] != 0);  //check that the pointers have been set up!

        storeJets(m_inputLeafCards[iLeaf]->getOutputJetsA(), iLeaf, 0);
        storeJets(m_inputLeafCards[iLeaf]->getOutputJetsB(), iLeaf, MAX_JETS_IN);
        storeJets(m_inputLeafCards[iLeaf]->getOutputJetsC(), iLeaf, 2*MAX_JETS_IN);
        
        // Deal with the Ht inputs
        // Fetch the output values from each of our input leaf cards.
        // Use setInputHt() to fill the inputEx[i] variables.
        setInputHt(iLeaf, m_inputLeafCards[iLeaf]->getOutputHt());        
    }
}

void L1GctWheelJetFpga::process()
{
    classifyJets();

    sort(m_rawCentralJets.begin(), m_rawCentralJets.end(), L1GctJetCand::rankGreaterThan());
    sort(m_rawForwardJets.begin(), m_rawForwardJets.end(), L1GctJetCand::rankGreaterThan());
    sort(m_rawTauJets.begin(), m_rawTauJets.end(), L1GctJetCand::rankGreaterThan());
    
    for(unsigned short iJet = 0; iJet < MAX_JETS_OUT; ++iJet)
    {
        m_centralJets[iJet] = m_rawCentralJets[iJet];
        m_forwardJets[iJet] = m_rawForwardJets[iJet];
        m_tauJets[iJet] = m_rawTauJets[iJet];
    }

    //For use with Ht processing        
    vector<int> htVal(3);
    unsigned long htSum;
    bool htOfl = false;
    int temp;

    // Deal with the Ht summing
    // Form the Ht sum from the inputs
    // sent from the Leaf cards
    for (int i = 0; i < MAX_LEAF_CARDS; ++i) 
    {
        // Decode input Ht value with overflow bit
        temp = (int) m_inputHt[i].to_ulong();
        if (temp>=Emax) 
        {
            htOfl = true;
            temp = temp % Emax;
        }
        htVal[i] = temp;
    }

    // Form Et sum taking care of overflows
    temp = htVal[0] + htVal[1] + htVal[2];
    if(temp>=Emax) 
    {
        htOfl = true;
        temp -= Emax;
    }
    htSum = (unsigned long) temp;

    // Convert output back to bitset format 
    bitset<NUM_BITS_ENERGY_DATA> htBits(htSum);
    if(htOfl) { htBits.set(OVERFLOW_BIT); }

    m_outputHt = htBits;
    
}

void L1GctWheelJetFpga::setInputLeafCard(int i, L1GctJetLeafCard* card)
{
    assert(i >= 0 && i < MAX_LEAF_CARDS);
    m_inputLeafCards[i] = card;
}

void L1GctWheelJetFpga::setInputJet(int i, L1GctJetCand jet)
{
    assert(i >=0 && i < MAX_JETS_IN);
    m_inputJets[i] =  jet;    
}

void L1GctWheelJetFpga::setInputHt (int i, unsigned ht)
{   
    unsigned long htVal;
    bool htOfl;

    int temp;

    if (i>=0 && i<MAX_LEAF_CARDS) 
    {
        // Transform the input variables into the correct range,
        // and set the overflow bit if they are out of range.
        // The correct range is between 0 to (Emax-1).
        // Then copy the inputs into an unsigned long variable.
        temp = ((int) ht) % Emax;
        htOfl = (temp != ((int) ht));
        htVal = (unsigned long) temp;

        // Copy the data into the internal bitset format.
        bitset<NUM_BITS_ENERGY_DATA> htBits(htVal);
        if (htOfl) {htBits.set(OVERFLOW_BIT);}

        m_inputHt[i] = htBits;
    }
} 

void L1GctWheelJetFpga::storeJets(JetVector jets, unsigned short iLeaf, unsigned short offset)
{
    for(unsigned short iJet = 0; iJet < L1GctJetFinder::MAX_JETS_OUT; ++iJet)
    {
        m_inputJets[iLeaf*MAX_JETS_IN/MAX_LEAF_CARDS + offset + iJet] = jets[iJet];
    }
}

void L1GctWheelJetFpga::classifyJets()
{    
    //Clear the contents of all three of the  raw jet vectors
    m_rawCentralJets.clear();
    m_rawForwardJets.clear();
    m_rawTauJets.clear();
    
    //Holds which jet finder we are on, in phi (from 0 to 8).
    unsigned short int jetFinderIndex = 0;
    
    //counter to help in calculation of jetFinderIndex.
    unsigned short int jetIndex = 0;
     
    JetVector::iterator currentJet;
     
    for(currentJet = m_inputJets.begin(); currentJet != m_inputJets.end(); ++currentJet)
    {
        if(currentJet->eta() >= L1GctJetCand::LOCAL_ETA_HF_START)  //forward jet
        {
            m_rawForwardJets.push_back(currentJet->convertToGlobalJet(jetFinderIndex, m_id));
        }
        else
        {
            if(currentJet->tauVeto() == true)  //central non-tau jet.
            {
                m_rawCentralJets.push_back(currentJet->convertToGlobalJet(jetFinderIndex, m_id));
            }
            else  //must be central tau-jet
            {
                m_rawTauJets.push_back(currentJet->convertToGlobalJet(jetFinderIndex, m_id));
            }
        }
        //move onto the next jet finder phi position every 6 jets
        if(++jetIndex % L1GctJetFinder::MAX_JETS_OUT == 0) { ++jetFinderIndex; }
    }
}

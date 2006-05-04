 #include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetFinder.h"

#include <iostream>
//#include <math.h>
using namespace std;

L1GctJetFinder::L1GctJetFinder(EtaHalf etaHalf):
m_sourceCards(maxSourceCards),
m_inputRegions(maxRegionsIn),
m_outputJets(maxJets),
m_etaHalf(etaHalf)
{
}

L1GctJetFinder::~L1GctJetFinder()
{
}

void L1GctJetFinder::reset()
{
    m_inputRegions.clear();
    m_inputRegions.resize(maxRegionsIn);
    m_outputJets.clear();
    m_outputJets.resize(maxJets);
    m_outputHt = 0;
}

// Can't see a better way of doing this mapping...
void L1GctJetFinder::fetchInput()
{
    vector<L1GctRegion> tempRegions;  //for temp local copy of region data
    
    switch(m_etaHalf)
    {
    case NEG_ETA_TYPE:
        tempRegions = m_sourceCards[0]->getRegions();   assert(tempRegions.size() > 11);
        m_inputRegions[27] = tempRegions[0];
        m_inputRegions[26] = tempRegions[1];
        m_inputRegions[13] = tempRegions[2];
        m_inputRegions[25] = tempRegions[3];
        m_inputRegions[20] = tempRegions[4];
        m_inputRegions[21] = tempRegions[5];
        m_inputRegions[22] = tempRegions[6];
        m_inputRegions[23] = tempRegions[7];
        m_inputRegions[32] = tempRegions[8];
        m_inputRegions[33] = tempRegions[9];
        m_inputRegions[34] = tempRegions[10];
        m_inputRegions[35] = tempRegions[11];
        
        tempRegions = m_sourceCards[1]->getRegions();   assert(tempRegions.size() > 9);
        m_inputRegions[19] = tempRegions[0];
        m_inputRegions[18] = tempRegions[1];
        m_inputRegions[17] = tempRegions[2];
        m_inputRegions[16] = tempRegions[3];
        m_inputRegions[15] = tempRegions[4];
        m_inputRegions[14] = tempRegions[5];
        m_inputRegions[31] = tempRegions[6];
        m_inputRegions[30] = tempRegions[7];
        m_inputRegions[29] = tempRegions[8];
        m_inputRegions[28] = tempRegions[9];
        
        tempRegions = m_sourceCards[2]->getRegions();   assert(tempRegions.size() > 11);
        m_inputRegions[3] = tempRegions[0];
        m_inputRegions[2] = tempRegions[1];
        m_inputRegions[1] = tempRegions[3];
        m_inputRegions[8] = tempRegions[8];
        m_inputRegions[9] = tempRegions[9];
        m_inputRegions[10] = tempRegions[10];
        m_inputRegions[11] = tempRegions[11];
        
        tempRegions = m_sourceCards[3]->getRegions();   assert(tempRegions.size() > 9);        
        m_inputRegions[7] = tempRegions[6];
        m_inputRegions[6] = tempRegions[7];
        m_inputRegions[5] = tempRegions[8];
        m_inputRegions[4] = tempRegions[9];
        
        tempRegions = m_sourceCards[4]->getRegions();   assert(tempRegions.size() > 6);
        m_inputRegions[0] = tempRegions[6];
        
        tempRegions = m_sourceCards[5]->getRegions();   assert(tempRegions.size() > 6);
        m_inputRegions[12] = tempRegions[0];
        m_inputRegions[24] = tempRegions[6];
        
        tempRegions = m_sourceCards[6]->getRegions();   assert(tempRegions.size() > 0);
        m_inputRegions[36] = tempRegions[0];
        
        tempRegions = m_sourceCards[7]->getRegions();   assert(tempRegions.size() > 7);
        m_inputRegions[37] = tempRegions[2];
        m_inputRegions[44] = tempRegions[4];
        m_inputRegions[45] = tempRegions[5];
        m_inputRegions[46] = tempRegions[6];
        m_inputRegions[47] = tempRegions[7];
        
        tempRegions = m_sourceCards[8]->getRegions();   assert(tempRegions.size() > 5);
        m_inputRegions[43] = tempRegions[0];
        m_inputRegions[42] = tempRegions[1];
        m_inputRegions[41] = tempRegions[2];
        m_inputRegions[40] = tempRegions[3];
        m_inputRegions[39] = tempRegions[4];
        m_inputRegions[38] = tempRegions[5];
        break;
        
    case POS_ETA_TYPE:
        tempRegions = m_sourceCards[0]->getRegions();   assert(tempRegions.size() > 11);
        m_inputRegions[29] = tempRegions[0];
        m_inputRegions[30] = tempRegions[1];
        m_inputRegions[19] = tempRegions[2];
        m_inputRegions[31] = tempRegions[3];
        m_inputRegions[20] = tempRegions[4];
        m_inputRegions[21] = tempRegions[5];
        m_inputRegions[22] = tempRegions[6];
        m_inputRegions[23] = tempRegions[7];
        m_inputRegions[32] = tempRegions[8];
        m_inputRegions[33] = tempRegions[9];
        m_inputRegions[34] = tempRegions[10];
        m_inputRegions[35] = tempRegions[11];
        
        tempRegions = m_sourceCards[1]->getRegions();   assert(tempRegions.size() > 9);
        m_inputRegions[13] = tempRegions[0];
        m_inputRegions[14] = tempRegions[1];
        m_inputRegions[15] = tempRegions[2];
        m_inputRegions[16] = tempRegions[3];
        m_inputRegions[17] = tempRegions[4];
        m_inputRegions[18] = tempRegions[5];
        m_inputRegions[25] = tempRegions[6];
        m_inputRegions[26] = tempRegions[7];
        m_inputRegions[27] = tempRegions[8];
        m_inputRegions[28] = tempRegions[9];
        
        tempRegions = m_sourceCards[2]->getRegions();   assert(tempRegions.size() > 11);        
        m_inputRegions[5] = tempRegions[0];
        m_inputRegions[6] = tempRegions[1];
        m_inputRegions[7] = tempRegions[3];
        m_inputRegions[8] = tempRegions[8];
        m_inputRegions[9] = tempRegions[9];
        m_inputRegions[10] = tempRegions[10];
        m_inputRegions[11] = tempRegions[11];
        
        tempRegions = m_sourceCards[3]->getRegions();   assert(tempRegions.size() > 9);
        m_inputRegions[1] = tempRegions[6];
        m_inputRegions[2] = tempRegions[7];
        m_inputRegions[3] = tempRegions[8];
        m_inputRegions[4] = tempRegions[9];
        
        tempRegions = m_sourceCards[4]->getRegions();   assert(tempRegions.size() > 3);
        m_inputRegions[0] = tempRegions[3];
        
        tempRegions = m_sourceCards[5]->getRegions();   assert(tempRegions.size() > 3);
        m_inputRegions[12] = tempRegions[2];
        m_inputRegions[24] = tempRegions[3];
        
        tempRegions = m_sourceCards[6]->getRegions();   assert(tempRegions.size() > 2);
        m_inputRegions[36] = tempRegions[2];
        
        tempRegions = m_sourceCards[7]->getRegions();   assert(tempRegions.size() > 7);
        m_inputRegions[43] = tempRegions[2];
        m_inputRegions[44] = tempRegions[4];
        m_inputRegions[45] = tempRegions[5];
        m_inputRegions[46] = tempRegions[6];
        m_inputRegions[47] = tempRegions[7];
        
        tempRegions = m_sourceCards[8]->getRegions();   assert(tempRegions.size() > 5);
        m_inputRegions[37] = tempRegions[0];
        m_inputRegions[38] = tempRegions[1];
        m_inputRegions[39] = tempRegions[2];
        m_inputRegions[40] = tempRegions[3];
        m_inputRegions[41] = tempRegions[4];
        m_inputRegions[42] = tempRegions[5];
        break;
    }
}

void L1GctJetFinder::setInputSourceCard(unsigned i, L1GctSourceCard* sc)
{
    assert(i >= 0 && i < maxSourceCards);
    m_sourceCards[i] = sc;
}

void L1GctJetFinder::setInputRegion(int i, L1GctRegion region)
{
    assert(i >= 0 && i < maxRegionsIn);
    m_inputRegions[i] = region;
}

// For STL sorting... binary predicate for sorting jet ranks
bool rankGreaterThan (L1GctJet jet1, L1GctJet jet2)
{
   return (jet1.getRank() > jet2.getRank());
}

void L1GctJetFinder::process() 
{
    UShort jetNum = 0; //holds the number of jets currently found
    for(UShort column = 1; column <=2; ++column)  //Find jets in the central search region
    {
        //don't include row zero as it is not in the search region
        for (UShort row = 1; row < columnOffset; ++row)  
        {
            //the region index of the center of our 3*3 window
            UShort centreIndex = (column*columnOffset) + row;
            
            //Determine if we are at edge of the forward HCAL or not (so need 3*2 window)
            bool boundary = (centreIndex % columnOffset == columnOffset-1);

            //debug checks for improper input indices
            assert(centreIndex % columnOffset != 0);  //Don't want the 4 regions from other half of detector
            assert(centreIndex >= columnOffset);  //Don't want the shared column to left of jet finding area
            assert(centreIndex < (maxRegionsIn - columnOffset)); //Don't want column to the right either
                        
            if(detectJet(centreIndex, boundary))
            {
                assert(jetNum < maxJets);
                
                m_outputJets[jetNum].setRank(calcJetRank(centreIndex, boundary));
                m_outputJets[jetNum].setEta(row-1);
                m_outputJets[jetNum].setPhi(column-1);
                m_outputJets[jetNum].setTauVeto(calcJetTauVeto(centreIndex,boundary));

                ++jetNum;
            }
        }
    }
    //presort the jets into decending order of energy
    sort(m_outputJets.begin(), m_outputJets.end(), rankGreaterThan);
    
    //calculate the Ht
    m_outputHt = calcHt();
    
    return;
}

// Returns true if region index is the centre of a jet. Set boundary = true if at edge of HCAL.
bool L1GctJetFinder::detectJet(const UShort centreIndex, const bool boundary) const
{
    ULong testEt = m_inputRegions[centreIndex].getEt();  // Get the energy of the central region
    
    if(!boundary)  //Not at boundary, so use 3*3 window of regions to determine if a jet
    {
        //Test if our region qualifies as a jet by comparing its energy with the energies
        //of the surrounding eight regions.  In the event of neighbouring regions with
        //identical energy, this will locate the jet in the upper-most, left-most region.
        if(testEt >  m_inputRegions[centreIndex-1-columnOffset].getEt() &&
           testEt >  m_inputRegions[centreIndex - columnOffset].getEt() &&
           testEt >  m_inputRegions[centreIndex+1-columnOffset].getEt() &&
           
           testEt >  m_inputRegions[centreIndex - 1].getEt() &&
           testEt >= m_inputRegions[centreIndex + 1].getEt() &&
           
           testEt >= m_inputRegions[centreIndex-1+columnOffset].getEt() &&
           testEt >= m_inputRegions[centreIndex + columnOffset].getEt() &&
           testEt >= m_inputRegions[centreIndex+1+columnOffset].getEt())
        {
            return true;
        }
    }
    else 
    {
        //...so only test surround 5 regions in our jet testing
        if(testEt >  m_inputRegions[centreIndex-1-columnOffset].getEt() &&
           testEt >  m_inputRegions[centreIndex - columnOffset].getEt() &&
           
           testEt >  m_inputRegions[centreIndex - 1].getEt() &&
           
           testEt >= m_inputRegions[centreIndex-1+columnOffset].getEt() &&
           testEt >= m_inputRegions[centreIndex + columnOffset].getEt())
        {
            return true;
        }
    }
    return false;           
}

// returns the energy sum (rank) of the nine regions centred (physically) about centreIndex
ULong L1GctJetFinder::calcJetRank(const UShort centreIndex, const bool boundary) const
{
    ULong energy = 0;
    
    if(!boundary)
    {
        for(int column = -1; column <= +1; ++column)
        {
            energy += m_inputRegions[centreIndex-1 + (column*columnOffset)].getEt() +
                      m_inputRegions[ centreIndex  + (column*columnOffset)].getEt() +
                      m_inputRegions[centreIndex+1 + (column*columnOffset)].getEt();
        }
    }
    else
    {
        for(int column = -1; column <= +1; ++column)
        {
            energy += m_inputRegions[centreIndex-1 + (column*columnOffset)].getEt() +
                      m_inputRegions[ centreIndex  + (column*columnOffset)].getEt();
        }
    }
    return convertToRank(energy);                                      
}

// returns the combined tauveto of the nine regions centred (physically) about centreIndex
bool L1GctJetFinder::calcJetTauVeto(const UShort centreIndex, const bool boundary) const
{
    bool partial[3] = {false, false, false};
    
    if(!boundary)
    {
        for(int column = -1; column <= +1; ++column)
        {
            partial[column+1] = m_inputRegions[centreIndex-1 + (column*columnOffset)].getTauVeto() ||
                                m_inputRegions[ centreIndex  + (column*columnOffset)].getTauVeto() ||
                                m_inputRegions[centreIndex+1 + (column*columnOffset)].getTauVeto();
        }
    }
    else
    {
        for(int column = -1; column <= +1; ++column)
        {
            partial[column+1] = m_inputRegions[centreIndex-1 + (column*columnOffset)].getTauVeto() ||
                                m_inputRegions[ centreIndex  + (column*columnOffset)].getTauVeto();
        }
    }
    return partial[0] || partial[1] || partial[2];
}

// Converts a jet energy into a calibrated rank. Somehow.
// Somewhat arbitrary at present...it currently converts 10 bits
// into 6 by dividing the energy int by 16 (no rounding up).
ULong L1GctJetFinder::convertToRank(const ULong energy) const
{
    const UShort energyBitSize = 10;
    const UShort rankBitSize = 6;
    const ULong maxEnergy = 0x400; //static_cast<ULong>(pow(2,energyBitSize));

    if(energy < maxEnergy)
    {
      return energy/0x10; //static_cast<ULong>(pow(2,energyBitSize-rankBitSize));
    }
    
    return maxEnergy-1;
}

// Calculates total calibrated energy in jets (Ht) sum
ULong L1GctJetFinder::calcHt() const
{    
    ULong ht = 0;
    for(UShort i=0; i < m_outputJets.size(); ++i)
    {
        ht += m_outputJets[i].getRank();
    }
    return ht;
}

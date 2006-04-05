 #include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetFinder.h"
//#include "CMSSW/FWCore/Utilities/interface/Exception.h"

#include <iostream>
#include <math.h>
using namespace std;

L1GctJetFinder::L1GctJetFinder()
{
    m_inputRegions.resize(maxRegionsIn);
}

L1GctJetFinder::~L1GctJetFinder()
{
}

void L1GctJetFinder::reset()
{
    m_inputRegions.clear();
    m_inputRegions.resize(maxRegionsIn);
    m_outputJets.clear();
    m_outputHt = 0;
}

void L1GctJetFinder::setInputRegion(int i, L1GctRegion region)
{
/*    //First check that i isn't out of range - do we want an exception, or just ignore out of range data?
    if(i > maxRegionsIn || i < 0)
    {
        throw cms::Exception("RangeError")
         << "Region " << i << " is outside jet finder input region range of 0 to "
         << maxRegionsIn << "\n";
    }
*/
    if(i >= 0 || i < maxRegionsIn)
    {
        m_inputRegions[i] = region;
    }
    else
    {
        cout << "\nWarning! Inputted Calorimeter Region " << i
             << " is outside jet finder input region range of 0 to " 
             << maxRegionsIn << ".  This data will be ignored." << endl;
    }
}

// For STL sorting... binary predicate for sorting jet ranks
bool rankGreaterThan (L1GctJet jet1, L1GctJet jet2)
{
   return (jet1.getRank() > jet2.getRank());
}

void L1GctJetFinder::process() 
{
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
                L1GctJet tempJet;
                tempJet.setRank(calcJetRank(centreIndex, boundary));
                tempJet.setEta(row-1);
                tempJet.setPhi(column-1);
                tempJet.setTauVeto(calcJetTauVeto(centreIndex,boundary));
                
                m_outputJets.push_back(tempJet);
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
    const ULong maxEnergy = static_cast<ULong>(pow(2,energyBitSize));

    if(energy < maxEnergy)
    {
        return energy/static_cast<ULong>(pow(2,energyBitSize-rankBitSize));
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

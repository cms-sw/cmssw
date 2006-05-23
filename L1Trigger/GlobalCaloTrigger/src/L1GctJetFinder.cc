#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetFinder.h"
 
#include "FWCore/Utilities/interface/Exception.h"  

#include <iostream>
using namespace std;

L1GctJetFinder::L1GctJetFinder(int id):
  m_id(id),
  m_sourceCards(MAX_SOURCE_CARDS),
  m_inputRegions(MAX_REGIONS_IN),
  m_outputJets(MAX_JETS_OUT)
{
  assert(m_id >=0 && m_id < 18);
}

L1GctJetFinder::~L1GctJetFinder()
{
}

std::ostream& operator << (std::ostream& os, const L1GctJetFinder& algo)
{
  os << "ALGO ID " << algo.m_id << std::endl;
  os << "No. of Source cards " << algo.m_sourceCards.size() << std::endl;
  for(unsigned i=0; i < algo.m_sourceCards.size(); i++)
    {
      if (algo.m_sourceCards[i]!=0) os << (*algo.m_sourceCards[i]); // These can be NULL!
    }
  os << "No. of input regions " << algo.m_inputRegions.size() << std::endl;
  for(unsigned i=0; i < algo.m_inputRegions.size(); i++)
    {
      os << algo.m_inputRegions[i]; 
    }
  os << "No. of output jets " << algo.m_outputJets.size() << std::endl;
  for(unsigned i=0; i < algo.m_outputJets.size(); i++)
    {
      os << algo.m_outputJets[i]; 
    }
  os << "Output Ht " << algo.m_outputHt << std::endl;

  return os;
}


void L1GctJetFinder::reset()
{
  m_inputRegions.clear();
  m_inputRegions.resize(MAX_REGIONS_IN);
  m_outputJets.clear();
  m_outputJets.resize(MAX_JETS_OUT);
  m_outputHt = 0;
}

// Can't see a better way of doing this mapping...
void L1GctJetFinder::fetchInput()
{
  vector<L1GctRegion> tempRegions;  //for temp local copy of region data
    
  //Debug check the pointers are set!
  assert(m_sourceCards[0] != 0);
  assert(m_sourceCards[1] != 0);
  assert(m_sourceCards[2] != 0);
  assert(m_sourceCards[3] != 0);
  assert(m_sourceCards[4] != 0);
  assert(m_sourceCards[5] != 0);
  assert(m_sourceCards[6] != 0);
  assert(m_sourceCards[7] != 0);
  assert(m_sourceCards[8] != 0);
  assert(m_sourceCards[9] != 0);
    
  tempRegions = m_sourceCards[0]->getRegions();
  assert(tempRegions.size() == 10);  //further pointer setup checks...
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
    
  tempRegions = m_sourceCards[1]->getRegions();
  assert(tempRegions.size() == 12);
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

  tempRegions = m_sourceCards[2]->getRegions();
  assert(tempRegions.size() == 10);
  m_inputRegions[1] = tempRegions[6];
  m_inputRegions[2] = tempRegions[7];
  m_inputRegions[3] = tempRegions[8];
  m_inputRegions[4] = tempRegions[9];
    
  tempRegions = m_sourceCards[3]->getRegions();
  assert(tempRegions.size() == 12);
  m_inputRegions[5] = tempRegions[0];
  m_inputRegions[6] = tempRegions[1];
  m_inputRegions[7] = tempRegions[3];
  m_inputRegions[8] = tempRegions[8];
  m_inputRegions[9] = tempRegions[9];
  m_inputRegions[10] = tempRegions[10];
  m_inputRegions[11] = tempRegions[11];
    
  tempRegions = m_sourceCards[4]->getRegions();
  assert(tempRegions.size() == 10);
  m_inputRegions[0] = tempRegions[6];
    
  tempRegions = m_sourceCards[5]->getRegions();
  assert(tempRegions.size() == 10);
  m_inputRegions[12] = tempRegions[0];
  m_inputRegions[24] = tempRegions[6];
    
  tempRegions = m_sourceCards[6]->getRegions();
  assert(tempRegions.size() == 10);
  m_inputRegions[36] = tempRegions[0];
    
  tempRegions = m_sourceCards[7]->getRegions();
  assert(tempRegions.size() == 10);
  m_inputRegions[37] = tempRegions[0];
  m_inputRegions[38] = tempRegions[1];
  m_inputRegions[39] = tempRegions[2];
  m_inputRegions[40] = tempRegions[3];
  m_inputRegions[41] = tempRegions[4];
  m_inputRegions[42] = tempRegions[5];
        
  tempRegions = m_sourceCards[8]->getRegions();
  assert(tempRegions.size() == 12);
  m_inputRegions[43] = tempRegions[2];
  m_inputRegions[44] = tempRegions[4];
  m_inputRegions[45] = tempRegions[5];
  m_inputRegions[46] = tempRegions[6];
  m_inputRegions[47] = tempRegions[7];
}
void L1GctJetFinder::setInputSourceCard(int i, L1GctSourceCard* sc)
{
  if(i >= 0 && i < MAX_SOURCE_CARDS)
  {
    m_sourceCards[i] = sc;
  }
  else
  {
    throw cms::Exception("L1GctSetupError")
    << "L1GctJetFinder::setInputSourceCard() : Source Card " << i << " is outside input range of 0 to "
    << (MAX_SOURCE_CARDS-1) << "\n";
  }
}

void L1GctJetFinder::setInputRegion(int i, L1GctRegion region)
{
  if(i >= 0 && i < MAX_REGIONS_IN)
  {
    m_inputRegions[i] = region;
  }
  else
  {
    throw cms::Exception("L1GctInputError")
    << "L1GctJetFinder::setInputRegion() : Region " << i << " is outside input range of 0 to "
    << (MAX_REGIONS_IN-1) << "\n";
  }
}

void L1GctJetFinder::process() 
{
  UShort jetNum = 0; //holds the number of jets currently found
  for(UShort column = 1; column <=2; ++column)  //Find jets in the central search region
  {
    //don't include row zero as it is not in the search region
    for (UShort row = 1; row < COL_OFFSET; ++row)  
    {
      //the region index of the center of our 3*3 window
      UShort centreIndex = (column*COL_OFFSET) + row;
           
      //Determine if we are at end of the HF or not (so need 3*2 window)
      bool hfBoundary = (row == COL_OFFSET-1);
      //Determine if we are at the end of the endcap HCAL regions, so need boundary condition tauveto
      bool heBoundary = (row == COL_OFFSET-5);

      //debug checks for improper input indices
      assert(centreIndex % COL_OFFSET != 0);  //Don't want the 4 regions from other half of detector
      assert(centreIndex >= COL_OFFSET);  //Don't want the shared column to left of jet finding area
      assert(centreIndex < (MAX_REGIONS_IN - COL_OFFSET)); //Don't want column to the right either
                        
      if(detectJet(centreIndex, hfBoundary))
      {
        assert(jetNum < MAX_JETS_OUT);
                
        m_outputJets[jetNum].setRank(calcJetRank(centreIndex, hfBoundary));
        m_outputJets[jetNum].setEta(row-1);
        m_outputJets[jetNum].setPhi(column-1);
        if(row < COL_OFFSET-4)  //if we are not in the HF, perform tauVeto analysis
        {
          m_outputJets[jetNum].setTauVeto(calcJetTauVeto(centreIndex,heBoundary));
        }
        else //can't be a tau jet because we are in the HF
        {
          m_outputJets[jetNum].setTauVeto(true);
        }
        ++jetNum;
      }
    }
  }
  //presort the jets into decending order of energy
  sort(m_outputJets.begin(), m_outputJets.end(), L1GctJetCand::rankGreaterThan());
   
  //calculate the Ht
  m_outputHt = calcHt();
    
  return;
}

// Returns true if region index is the centre of a jet. Set boundary = true if at edge of HCAL.
bool L1GctJetFinder::detectJet(const UShort centreIndex, const bool boundary) const
{
  if(!boundary)  //Not at boundary, so use 3*3 window of regions to determine if a jet
  {
    // Get the energy of the central region
    ULong testEt = m_inputRegions[centreIndex].getEt();
        
    //Test if our region qualifies as a jet by comparing its energy with the energies of the
    //surrounding eight regions.  In the event of neighbouring regions with identical energy,
    //this will locate the jet in the lower-most (furthest away from eta=0), left-most (least phi) region.
    if(testEt >  m_inputRegions[centreIndex-1-COL_OFFSET].getEt() &&
       testEt >  m_inputRegions[centreIndex - COL_OFFSET].getEt() &&
       testEt >  m_inputRegions[centreIndex+1-COL_OFFSET].getEt() &&
           
       testEt >= m_inputRegions[centreIndex - 1].getEt() &&
       testEt >  m_inputRegions[centreIndex + 1].getEt() &&
           
       testEt >= m_inputRegions[centreIndex-1+COL_OFFSET].getEt() &&
       testEt >= m_inputRegions[centreIndex + COL_OFFSET].getEt() &&
       testEt >= m_inputRegions[centreIndex+1+COL_OFFSET].getEt())
    {
      return true;
    }
//USE THIS BLOCK INSTEAD IF YOU WANT OVERFLOW BIT FUNCTIONALITY        
/*    // Get the energy of the central region & OR the overflow bit to become the MSB
    ULong testEt = (m_inputRegions[centreIndex].getEt() | (m_inputRegions[centreIndex].getOverFlow() << L1GctRegion::ET_BITWIDTH));
        
    //Test if our region qualifies as a jet by comparing its energy with the energies of the
    //surrounding eight regions.  In the event of neighbouring regions with identical energy,
    //this will locate the jet in the lower-most (furthest away from eta=0), left-most (least phi) region.
    if(testEt >  (m_inputRegions[centreIndex-1-COL_OFFSET].getEt() | (m_inputRegions[centreIndex-1-COL_OFFSET].getOverFlow() << L1GctRegion::ET_BITWIDTH)) &&
       testEt >  (m_inputRegions[centreIndex - COL_OFFSET].getEt() | (m_inputRegions[centreIndex - COL_OFFSET].getOverFlow() << L1GctRegion::ET_BITWIDTH)) &&
       testEt >  (m_inputRegions[centreIndex+1-COL_OFFSET].getEt() | (m_inputRegions[centreIndex+1-COL_OFFSET].getOverFlow() << L1GctRegion::ET_BITWIDTH)) &&
           
       testEt >= (m_inputRegions[centreIndex - 1].getEt() | (m_inputRegions[centreIndex - 1].getOverFlow() << L1GctRegion::ET_BITWIDTH)) &&
       testEt >  (m_inputRegions[centreIndex + 1].getEt() | (m_inputRegions[centreIndex + 1].getOverFlow() << L1GctRegion::ET_BITWIDTH)) &&
           
       testEt >= (m_inputRegions[centreIndex-1+COL_OFFSET].getEt() | (m_inputRegions[centreIndex-1+COL_OFFSET].getOverFlow() << L1GctRegion::ET_BITWIDTH)) &&
       testEt >= (m_inputRegions[centreIndex + COL_OFFSET].getEt() | (m_inputRegions[centreIndex + COL_OFFSET].getOverFlow() << L1GctRegion::ET_BITWIDTH)) &&
       testEt >= (m_inputRegions[centreIndex+1+COL_OFFSET].getEt() | (m_inputRegions[centreIndex+1+COL_OFFSET].getOverFlow() << L1GctRegion::ET_BITWIDTH)))
    {
      return true;
    }
*/  //END OVERFLOW FUNCTIONALITY       
  }
  else    //...so only test surround 5 regions in our jet testing.
  {    
    // Get the energy of the central region
    // Don't need all the overflow bit adjustments as above, since we are in the HF here
    ULong testEt = m_inputRegions[centreIndex].getEt();        
        
    if(testEt >  m_inputRegions[centreIndex-1-COL_OFFSET].getEt() &&
       testEt >  m_inputRegions[centreIndex - COL_OFFSET].getEt() &&
       
       testEt >= m_inputRegions[centreIndex - 1].getEt() &&
           
       testEt >= m_inputRegions[centreIndex-1+COL_OFFSET].getEt() &&
       testEt >= m_inputRegions[centreIndex + COL_OFFSET].getEt())
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
      energy += m_inputRegions[centreIndex-1 + (column*COL_OFFSET)].getEt() +
                m_inputRegions[ centreIndex  + (column*COL_OFFSET)].getEt() +
                m_inputRegions[centreIndex+1 + (column*COL_OFFSET)].getEt();
    }
  }
  else
  {
    for(int column = -1; column <= +1; ++column)
    {
      energy += m_inputRegions[centreIndex-1 + (column*COL_OFFSET)].getEt() +
                m_inputRegions[ centreIndex  + (column*COL_OFFSET)].getEt();
    }
  }
  return convertToRank(energy);                                      
}

// returns the combined tauveto of the nine regions centred (physically) about centreIndex. Set boundary = true if at edge of Endcap.
bool L1GctJetFinder::calcJetTauVeto(const UShort centreIndex, const bool boundary) const
{
  bool partial[3] = {false, false, false};
    
  if(!boundary)
  {
    for(int column = -1; column <= +1; ++column)
    {
      partial[column+1] = m_inputRegions[centreIndex-1 + (column*COL_OFFSET)].getTauVeto() ||
                          m_inputRegions[ centreIndex  + (column*COL_OFFSET)].getTauVeto() ||
                          m_inputRegions[centreIndex+1 + (column*COL_OFFSET)].getTauVeto();
    }
  }
  else
  {
    for(int column = -1; column <= +1; ++column)
    {
      partial[column+1] = m_inputRegions[centreIndex-1 + (column*COL_OFFSET)].getTauVeto() ||
                          m_inputRegions[ centreIndex  + (column*COL_OFFSET)].getTauVeto();
    }
  }
  return partial[0] || partial[1] || partial[2];
}

// Converts a jet energy into a calibrated rank. Somehow.
// Somewhat arbitrary at present...it currently converts 10 bits
// into 6 by dividing the energy int by 16 (no rounding up).
ULong L1GctJetFinder::convertToRank(const ULong energy) const
{
  if(energy < 1024)
  {
    return energy/16;
  }
    
  return 63;
}

// Calculates total calibrated energy in jets (Ht) sum
L1GctScalarEtVal L1GctJetFinder::calcHt() const
{    
  unsigned ht = 0;
  for(UShort i=0; i < MAX_JETS_OUT; ++i)
  {
    ht += m_outputJets[i].rank();
  }
  L1GctScalarEtVal temp(ht);
  return temp;
}

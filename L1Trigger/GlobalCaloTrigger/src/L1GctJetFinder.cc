#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetFinder.h"
 
#include "FWCore/Utilities/interface/Exception.h"  

#include <iostream>
using namespace std;

//DEFINE STATICS
const int L1GctJetFinder::MAX_JETS_OUT = 6;
const unsigned int L1GctJetFinder::MAX_SOURCE_CARDS = 9;
const int L1GctJetFinder::COL_OFFSET = ((L1GctMap::N_RGN_ETA)/2)+1;
const int L1GctJetFinder::MAX_REGIONS_IN = L1GctJetFinder::COL_OFFSET*4;


L1GctJetFinder::L1GctJetFinder(int id, vector<L1GctSourceCard*> sourceCards,
                               L1GctJetEtCalibrationLut* jetEtCalLut):
  m_id(id),
  m_sourceCards(sourceCards),
  m_jetEtCalLut(jetEtCalLut),
  m_inputRegions(MAX_REGIONS_IN),
  m_outputJets(MAX_JETS_OUT)
{
  //Check jetfinder setup
  if(m_id < 0 || m_id > 17)
  {
    throw cms::Exception("L1GctSetupError")
    << "L1GctJetFinder::L1GctJetFinder() : Jet Finder ID " << m_id << " has been incorrectly constructed!\n"
    << "ID number should be between the range of 0 to 17\n";
  } 
  
  if(m_sourceCards.size() != MAX_SOURCE_CARDS)
  {
    throw cms::Exception("L1GctSetupError")
    << "L1GctJetFinder::L1GctJetFinder() : Jet Finder ID " << m_id << " has been incorrectly constructed!\n"
    << "This class needs " << MAX_SOURCE_CARDS << " source card pointers, yet only " << m_sourceCards.size()
    << " source card pointers are present.\n";
  }
  
  for(unsigned int i = 0; i < m_sourceCards.size(); ++i)
  {
    if(m_sourceCards.at(i) == 0)
    {
      throw cms::Exception("L1GctSetupError")
      << "L1GctJetFinder::L1GctJetFinder() : Jet Finder ID " << m_id << " has been incorrectly constructed!\n"
      << "Source card pointer " << i << " has not been set!\n";
    }
  }
  
  if(m_jetEtCalLut == 0)
  {
    throw cms::Exception("L1GctSetupError")
    << "L1GctJetFinder::L1GctJetFinder() : Jet Finder ID " << m_id << " has been incorrectly constructed!\n"
    << "The jet Et calibration LUT pointer has not been set!\n";  
  }
}

L1GctJetFinder::~L1GctJetFinder()
{
}

ostream& operator << (ostream& os, const L1GctJetFinder& algo)
{
  os << "===L1GctJetFinder===" << endl;
  os << "ID = " << algo.m_id << endl;
  os << "No of Source cards " << algo.m_sourceCards.size() << endl;
  for (unsigned i=0; i<algo.m_sourceCards.size(); i++) {
    os << "SourceCard* " << i << " = " << algo.m_sourceCards.at(i)<< endl;
    os << "No of regions from this sourceCard " << algo.m_sourceCards.at(i)->getRegions().size() << endl;
  }
  os << "JetEtCalibrationLut* = " <<  algo.m_jetEtCalLut << endl;
  os << "No of input regions " << algo.m_inputRegions.size() << endl;
//   for(unsigned i=0; i < algo.m_inputRegions.size(); ++i)
//     {
//       os << algo.m_inputRegions.at(i); 
//     }
  os << "No of output jets " << algo.m_outputJets.size() << endl;
//   for(unsigned i=0; i < algo.m_outputJets.size(); ++i)
//     {
//       os << algo.m_outputJets.at(i); 
//     }
  os << "Output Et strip 0 " << algo.m_outputEtStrip0 << endl;
  os << "Output Et strip 1 " << algo.m_outputEtStrip1 << endl;
  os << "Output Ht " << algo.m_outputHt << endl;
  os << endl;

  return os;
}


void L1GctJetFinder::reset()
{
  m_inputRegions.clear();
  m_inputRegions.resize(MAX_REGIONS_IN);
  m_outputJets.clear();
  m_outputJets.resize(MAX_JETS_OUT);
  m_outputEtStrip0 = 0;
  m_outputEtStrip1 = 0;
  m_outputHt = 0;
}

// Can't see a better way of doing this mapping...
void L1GctJetFinder::fetchInput()
{
  vector<L1GctRegion> tempRegions;  //for temp local copy of region data
    
  tempRegions = m_sourceCards.at(0)->getRegions();
  assert(tempRegions.size() == 10);  //Pointer setup check...
  m_inputRegions.at(13) = tempRegions.at(0);
  m_inputRegions.at(14) = tempRegions.at(1);
  m_inputRegions.at(15) = tempRegions.at(2);
  m_inputRegions.at(16) = tempRegions.at(3);
  m_inputRegions.at(17) = tempRegions.at(4);
  m_inputRegions.at(18) = tempRegions.at(5);
  m_inputRegions.at(25) = tempRegions.at(6);
  m_inputRegions.at(26) = tempRegions.at(7);
  m_inputRegions.at(27) = tempRegions.at(8);
  m_inputRegions.at(28) = tempRegions.at(9);
    
  tempRegions = m_sourceCards.at(1)->getRegions();
  assert(tempRegions.size() == 12);
  m_inputRegions.at(29) = tempRegions.at(0);
  m_inputRegions.at(30) = tempRegions.at(1);
  m_inputRegions.at(19) = tempRegions.at(2);
  m_inputRegions.at(31) = tempRegions.at(3);
  m_inputRegions.at(20) = tempRegions.at(4);
  m_inputRegions.at(21) = tempRegions.at(5);
  m_inputRegions.at(22) = tempRegions.at(6);
  m_inputRegions.at(23) = tempRegions.at(7);
  m_inputRegions.at(32) = tempRegions.at(8);
  m_inputRegions.at(33) = tempRegions.at(9);
  m_inputRegions.at(34) = tempRegions.at(10);
  m_inputRegions.at(35) = tempRegions.at(11);

  tempRegions = m_sourceCards.at(2)->getRegions();
  assert(tempRegions.size() == 10);
  m_inputRegions.at(1) = tempRegions.at(6);
  m_inputRegions.at(2) = tempRegions.at(7);
  m_inputRegions.at(3) = tempRegions.at(8);
  m_inputRegions.at(4) = tempRegions.at(9);
    
  tempRegions = m_sourceCards.at(3)->getRegions();
  assert(tempRegions.size() == 12);
  m_inputRegions.at(5) = tempRegions.at(0);
  m_inputRegions.at(6) = tempRegions.at(1);
  m_inputRegions.at(7) = tempRegions.at(3);
  m_inputRegions.at(8) = tempRegions.at(8);
  m_inputRegions.at(9) = tempRegions.at(9);
  m_inputRegions.at(10) = tempRegions.at(10);
  m_inputRegions.at(11) = tempRegions.at(11);
    
  tempRegions = m_sourceCards.at(4)->getRegions();
  assert(tempRegions.size() == 10);
  m_inputRegions.at(0) = tempRegions.at(6);
    
  tempRegions = m_sourceCards.at(5)->getRegions();
  assert(tempRegions.size() == 10);
  m_inputRegions.at(12) = tempRegions.at(0);
  m_inputRegions.at(24) = tempRegions.at(6);
    
  tempRegions = m_sourceCards.at(6)->getRegions();
  assert(tempRegions.size() == 10);
  m_inputRegions.at(36) = tempRegions.at(0);
    
  tempRegions = m_sourceCards.at(7)->getRegions();
  assert(tempRegions.size() == 10);
  m_inputRegions.at(37) = tempRegions.at(0);
  m_inputRegions.at(38) = tempRegions.at(1);
  m_inputRegions.at(39) = tempRegions.at(2);
  m_inputRegions.at(40) = tempRegions.at(3);
  m_inputRegions.at(41) = tempRegions.at(4);
  m_inputRegions.at(42) = tempRegions.at(5);
        
  tempRegions = m_sourceCards.at(8)->getRegions();
  assert(tempRegions.size() == 12);
  m_inputRegions.at(43) = tempRegions.at(2);
  m_inputRegions.at(44) = tempRegions.at(4);
  m_inputRegions.at(45) = tempRegions.at(5);
  m_inputRegions.at(46) = tempRegions.at(6);
  m_inputRegions.at(47) = tempRegions.at(7);
}


void L1GctJetFinder::setInputRegion(int i, L1GctRegion region)
{
  if(i >= 0 && i < MAX_REGIONS_IN)
  {
    m_inputRegions.at(i) = region;
  }
  else
  {
    throw cms::Exception("L1GctInputError")
    << "L1GctJetFinder::setInputRegion() : In Jet Finder ID " << m_id << ", inputted region " 
    << i << " is outside input index range of 0 to " << (MAX_REGIONS_IN-1) << "\n";
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
                
        m_outputJets.at(jetNum).setRank(m_jetEtCalLut->convertToSixBitRank(static_cast<uint16_t>(calcJetEnergy(centreIndex, hfBoundary)), (row-1)));
        m_outputJets.at(jetNum).setEta(row-1);
        m_outputJets.at(jetNum).setPhi(column-1);
        if(row < COL_OFFSET-4)  //if we are not in the HF, perform tauVeto analysis
        {
          m_outputJets.at(jetNum).setTauVeto(calcJetTauVeto(centreIndex,heBoundary));
        }
        else //can't be a tau jet because we are in the HF
        {
          m_outputJets.at(jetNum).setTauVeto(true);
        }
        ++jetNum;
      }
    }
  }
  //presort the jets into decending order of energy
  sort(m_outputJets.begin(), m_outputJets.end(), L1GctJet::rankGreaterThan());
   
  //calculate the raw Et strip sums
  m_outputEtStrip0 = calcEtStrip(0);
  m_outputEtStrip1 = calcEtStrip(1);

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
    ULong testEt = m_inputRegions.at(centreIndex).et();
        
    //Test if our region qualifies as a jet by comparing its energy with the energies of the
    //surrounding eight regions.  In the event of neighbouring regions with identical energy,
    //this will locate the jet in the lower-most (furthest away from eta=0), left-most (least phi) region.
    if(testEt >  m_inputRegions.at(centreIndex-1-COL_OFFSET).et() &&
       testEt >  m_inputRegions.at(centreIndex - COL_OFFSET).et() &&
       testEt >  m_inputRegions.at(centreIndex+1-COL_OFFSET).et() &&
           
       testEt >= m_inputRegions.at(centreIndex - 1).et() &&
       testEt >  m_inputRegions.at(centreIndex + 1).et() &&
           
       testEt >= m_inputRegions.at(centreIndex-1+COL_OFFSET).et() &&
       testEt >= m_inputRegions.at(centreIndex + COL_OFFSET).et() &&
       testEt >= m_inputRegions.at(centreIndex+1+COL_OFFSET).et())
    {
      return true;
    }
//USE THIS BLOCK INSTEAD IF YOU WANT OVERFLOW BIT FUNCTIONALITY        
/*    // Get the energy of the central region & OR the overflow bit to become the MSB
    ULong testEt = (m_inputRegions.at(centreIndex).et() | (m_inputRegions.at(centreIndex).getOverFlow() << L1GctRegion::ET_BITWIDTH));
        
    //Test if our region qualifies as a jet by comparing its energy with the energies of the
    //surrounding eight regions.  In the event of neighbouring regions with identical energy,
    //this will locate the jet in the lower-most (furthest away from eta=0), left-most (least phi) region.
    if(testEt >  (m_inputRegions.at(centreIndex-1-COL_OFFSET).et() | (m_inputRegions.at(centreIndex-1-COL_OFFSET).getOverFlow() << L1GctRegion::ET_BITWIDTH)) &&
       testEt >  (m_inputRegions.at(centreIndex - COL_OFFSET).et() | (m_inputRegions.at(centreIndex - COL_OFFSET).getOverFlow() << L1GctRegion::ET_BITWIDTH)) &&
       testEt >  (m_inputRegions.at(centreIndex+1-COL_OFFSET).et() | (m_inputRegions.at(centreIndex+1-COL_OFFSET).getOverFlow() << L1GctRegion::ET_BITWIDTH)) &&
           
       testEt >= (m_inputRegions.at(centreIndex - 1).et() | (m_inputRegions.at(centreIndex - 1).getOverFlow() << L1GctRegion::ET_BITWIDTH)) &&
       testEt >  (m_inputRegions.at(centreIndex + 1).et() | (m_inputRegions.at(centreIndex + 1).getOverFlow() << L1GctRegion::ET_BITWIDTH)) &&
           
       testEt >= (m_inputRegions.at(centreIndex-1+COL_OFFSET).et() | (m_inputRegions.at(centreIndex-1+COL_OFFSET).getOverFlow() << L1GctRegion::ET_BITWIDTH)) &&
       testEt >= (m_inputRegions.at(centreIndex + COL_OFFSET).et() | (m_inputRegions.at(centreIndex + COL_OFFSET).getOverFlow() << L1GctRegion::ET_BITWIDTH)) &&
       testEt >= (m_inputRegions.at(centreIndex+1+COL_OFFSET).et() | (m_inputRegions.at(centreIndex+1+COL_OFFSET).getOverFlow() << L1GctRegion::ET_BITWIDTH)))
    {
      return true;
    }
*/  //END OVERFLOW FUNCTIONALITY       
  }
  else    //...so only test surround 5 regions in our jet testing.
  {    
    // Get the energy of the central region
    // Don't need all the overflow bit adjustments as above, since we are in the HF here
    ULong testEt = m_inputRegions.at(centreIndex).et();        
        
    if(testEt >  m_inputRegions.at(centreIndex-1-COL_OFFSET).et() &&
       testEt >  m_inputRegions.at(centreIndex - COL_OFFSET).et() &&
       
       testEt >= m_inputRegions.at(centreIndex - 1).et() &&
           
       testEt >= m_inputRegions.at(centreIndex-1+COL_OFFSET).et() &&
       testEt >= m_inputRegions.at(centreIndex + COL_OFFSET).et())
    {
      return true;
    }
  }
  return false;           
}

// returns the energy sum of the nine regions centred (physically) about centreIndex
ULong L1GctJetFinder::calcJetEnergy(const UShort centreIndex, const bool boundary) const
{
  ULong energy = 0;
    
  if(!boundary)
  {
    for(int column = -1; column <= +1; ++column)
    {
      energy += m_inputRegions.at(centreIndex-1 + (column*COL_OFFSET)).et() +
                m_inputRegions.at( centreIndex  + (column*COL_OFFSET)).et() +
                m_inputRegions.at(centreIndex+1 + (column*COL_OFFSET)).et();
    }
  }
  else
  {
    for(int column = -1; column <= +1; ++column)
    {
      energy += m_inputRegions.at(centreIndex-1 + (column*COL_OFFSET)).et() +
                m_inputRegions.at( centreIndex  + (column*COL_OFFSET)).et();
    }
  }
  return energy;                                   
}

// returns the combined tauveto of the nine regions centred (physically) about centreIndex. Set boundary = true if at edge of Endcap.
bool L1GctJetFinder::calcJetTauVeto(const UShort centreIndex, const bool boundary) const
{
  bool partial[3] = {false, false, false};
    
  if(!boundary)
  {
    for(int column = -1; column <= +1; ++column)
    {
      partial[column+1] = m_inputRegions.at(centreIndex-1 + (column*COL_OFFSET)).tauVeto() ||
                          m_inputRegions.at( centreIndex  + (column*COL_OFFSET)).tauVeto() ||
                          m_inputRegions.at(centreIndex+1 + (column*COL_OFFSET)).tauVeto();
    }
  }
  else
  {
    for(int column = -1; column <= +1; ++column)
    {
      partial[column+1] = m_inputRegions.at(centreIndex-1 + (column*COL_OFFSET)).tauVeto() ||
                          m_inputRegions.at( centreIndex  + (column*COL_OFFSET)).tauVeto();
    }
  }
  return partial[0] || partial[1] || partial[2];
}

// Calculates total (raw) energy in a phi strip
L1GctScalarEtVal L1GctJetFinder::calcEtStrip(const UShort strip) const
{
  if (strip !=0 && strip != 1) {
    throw cms::Exception("L1GctProcessingError")
      << "L1GctJetFinder::calcEtStrip() has been called with strip number "
      << strip << "; should be 0 or 1 \n";
  } 
  // Add the Et values from regions 13 to 23 for strip 0,
  //     the Et values from regions 25 to 35 for strip 1.
  unsigned et = 0;
  bool of = false;
  for (UShort i=1; i < COL_OFFSET; ++i) {
    et += m_inputRegions.at((COL_OFFSET*(strip+1)+i)).et();
    of |= m_inputRegions.at((COL_OFFSET*(strip+1)+i)).overFlow();
  }
  L1GctScalarEtVal temp(et);
  temp.setOverFlow(temp.overFlow() || of);
  return temp;
}

// Calculates total calibrated energy in jets (Ht) sum
L1GctScalarEtVal L1GctJetFinder::calcHt() const
{    
  unsigned ht = 0;
  for(UShort i=0; i < MAX_JETS_OUT; ++i)
  {
    // Only sum Ht for valid jets
    if (!m_outputJets.at(i).isNullJet()) {
      UShort eta = static_cast<UShort>(m_outputJets.at(i).eta());
      UShort phi = static_cast<UShort>(m_outputJets.at(i).phi());

      UShort centreIndex = ((phi+1)*COL_OFFSET) + (eta+1);
      bool hfBoundary = (eta == (COL_OFFSET-2));
      ht += m_jetEtCalLut->convertToTenBitRank(static_cast<uint16_t>(calcJetEnergy(centreIndex, hfBoundary)), eta);
    }
  }
  L1GctScalarEtVal temp(ht);
  return temp;
}

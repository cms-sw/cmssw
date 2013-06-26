// $Id: RPCData.cc,v 1.4 2013/03/20 15:45:25 wdd Exp $
// Include files 



// local
#include "L1Trigger/RPCTechnicalTrigger/interface/RPCData.h"

//-----------------------------------------------------------------------------
// Implementation file for class : RPCData
//
// 2008-11-18 : Andres Osorio
//-----------------------------------------------------------------------------

//=============================================================================
// Standard constructor, initializes variables
//=============================================================================

namespace l1trigger {

  Counters::Counters(int wheel)
  {
    m_wheelid = wheel;
    int maxsectors=12;
  
    for(int k=1; k <= maxsectors; ++k)
      m_sector[k] = 0;
    m_nearSide = 0;
    m_farSide = 0;
    m_wheel=0;
  
  }

  Counters::~Counters()
  {
    m_sector.clear();
  }

  void Counters::evalCounters()
  {
  
    std::map<int,int>::iterator itr;
    for(itr = m_sector.begin(); itr != m_sector.end(); ++itr)
      m_wheel += (*itr).second;
  
    std::vector<int> far;
    std::vector<int> near;
  
    far.push_back(3);
    far.push_back(4);
    far.push_back(5);
    far.push_back(6);
    far.push_back(7);
    far.push_back(8);
  
    near.push_back(1);
    near.push_back(2);
    near.push_back(12);
    near.push_back(11);
    near.push_back(10);
    near.push_back(9);
  
    std::vector<int>::iterator sec;
    for( sec = far.begin(); sec != far.end(); ++sec) {
      std::map<int, int>::iterator sector;
      sector = m_sector.find( (*sec) );
      m_farSide  += (*sector).second;
    }
  
    for( sec = near.begin(); sec != near.end(); ++sec) {
      std::map<int, int>::iterator sector;
      sector = m_sector.find( (*sec) );
      m_nearSide += (*sector).second;
    }
  

  }

  void Counters::printSummary()
  {

    std::cout << m_wheelid << std::endl;
    std::map<int,int>::iterator itr;
    for(itr = m_sector.begin(); itr != m_sector.end(); ++itr)
      std::cout << (*itr).first << ": " << (*itr).second << '\t';
    std::cout << '\n';
  
    std::cout << "total wheel: " 
              << m_wheel << " " << m_farSide << " " << m_nearSide << '\n';
  }

  void Counters::incrementSector( int sector )
  {
    m_sector[ sector ] += 1;
  }
}

RPCData::RPCData() {
  
  m_wheel     = 10;
  m_sec1      = new int[6];
  m_sec2      = new int[6];
  m_orsignals = new RBCInput[6];

}
//=============================================================================
// Destructor
//=============================================================================
RPCData::~RPCData() {
  
  delete [] m_sec1;
  delete [] m_sec2;
  delete [] m_orsignals;
  
} 

//=============================================================================

std::istream& operator>>(std::istream &istr , RPCData & rhs) 
{
  
  (istr) >> rhs.m_wheel;
  for(int k=0; k < 6; ++k)
  {
    (istr) >> rhs.m_sec1[k] >> rhs.m_sec2[k];
    (istr) >> rhs.m_orsignals[k];
  }

  return istr;
    
}

std::ostream& operator<<(std::ostream& ostr , RPCData & rhs) 
{
  
  ostr << rhs.m_wheel << '\t';
  for(int k=0; k < 6; ++k)
  {
    ostr << rhs.m_sec1[k] << '\t' <<  rhs.m_sec2[k] << '\n';
    ostr << rhs.m_orsignals[k];
  }
  
  return ostr;
  
}

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctSourceCard.h"

#include "FWCore/Utilities/interface/Exception.h"

#include <iostream>

using namespace std;

const int L1GctSourceCard::NUM_ELEC = 4;
const int L1GctSourceCard::NUM_REG_TYPE2 = 12; 
const int L1GctSourceCard::NUM_REG_TYPE3 = 10; 
const int L1GctSourceCard::N_MIP_BITS = 14;  
const int L1GctSourceCard::N_QUIET_BITS = 14; 

L1GctSourceCard::L1GctSourceCard(int id, SourceCardType typeVal):
  m_id(id),
  m_cardType(typeVal),
  m_mipBits(0),
  m_quietBits(0)
{
    this->setVectorSizes();
}

L1GctSourceCard::~L1GctSourceCard()
{
}

ostream& operator << (ostream& os, const L1GctSourceCard& card)
{
  os << "===L1GctSourceCard===" << endl;
  os << "ID " << card.m_id;
  os << " Type " << card.m_cardType;
  os << "No. of IsoElec " << card.m_isoElectrons.size() << endl;
  for(uint i=0; i < card.m_isoElectrons.size(); i++)
    {
      os << card.m_isoElectrons.at(i);
    } 
  os << "No. of NonIsoElec " << card.m_nonIsoElectrons.size() << endl;
  for(uint i=0; i < card.m_nonIsoElectrons.size(); i++)
    {
      os << card.m_nonIsoElectrons.at(i);
    }
  os << "MIPS " << card.m_mipBits;
  os << " QUIET " << card.m_quietBits << endl;
  os << "No. of Regions " << card.m_regions.size() << endl;
  for(uint i=0; i < card.m_regions.size(); i++)
    {
      os << card.m_regions.at(i);
    }
  return os;
}

void L1GctSourceCard::reset()
{
  // Clear the electron vectors
  m_isoElectrons.clear();
  m_nonIsoElectrons.clear();
    
  // Set the size of the electron vectors
  m_isoElectrons.resize(NUM_ELEC);
  m_nonIsoElectrons.resize(NUM_ELEC);
   
  m_mipBits=0;
  m_quietBits=0;
  // Don't clear and resize the regions vector.
  // Instead we reset all the elements.
  for (unsigned i=0; i<m_regions.size(); ++i) {
    m_regions.at(i).reset();
  }

}

/// set the Regions
void L1GctSourceCard::setRegions(vector<L1CaloRegion> regions) {
  if (m_cardType!=cardType2 && m_cardType!=cardType3) {
    throw cms::Exception("L1GctSetupError")
      << "L1GctSourceCard::setRegions() : Source Card ID: " << m_id << " is of cardType " << m_cardType << endl
      << "and only handles EM candidates" << endl;
  }
  else {
    if (m_cardType==cardType2 && regions.size()==12) {
      m_regions = regions;
    }
    else if (m_cardType==cardType3 && regions.size()==10) {
      m_regions = regions;
    }
    else {
    throw cms::Exception("L1GctSetupError")
      << "L1GctSourceCard::setRegions() : Source Card ID: " << m_id << " is of cardType " << m_cardType << endl
      << "Wrong number of regions. Tried to set " << regions.size() << " regions" << endl;
    }
  }
}

/// set the Iso Em candidates
void L1GctSourceCard::setIsoEm(vector<L1CaloEmCand> isoEm) {
  if (m_cardType!=cardType1) {
    throw cms::Exception("L1GctSetupError")
      << "L1GctSourceCard::setIsoEm() : Source Card ID: " << m_id << " is of cardType " << m_cardType << endl
      << "and only handles regions" << endl;
  }
  else {
    if (isoEm.size()==4) {
      m_isoElectrons = isoEm;
    }
    else {
    throw cms::Exception("L1GctSetupError")
      << "L1GctSourceCard::setIsoEm() : Source Card ID: " << m_id << " is of cardType " << m_cardType << endl
      << "Wrong number of electrons. Tried to set " << isoEm.size() << " electrons" << endl;
    }
  }
}

/// set the Non Iso Em candidates
void L1GctSourceCard::setNonIsoEm(vector<L1CaloEmCand> nonIsoEm) {
  if (m_cardType!=cardType1) {
    throw cms::Exception("L1GctSetupError")
      << "L1GctSourceCard::setNonIsoEm() : Source Card ID: " << m_id << " is of cardType " << m_cardType << endl
      << "and only handles regions" << endl;
  }
  else {
    if (nonIsoEm.size()==4) {
      m_nonIsoElectrons = nonIsoEm;
    }
    else {
    throw cms::Exception("L1GctSetupError")
      << "L1GctSourceCard::setNonIsoEm() : Source Card ID: " << m_id << " is of cardType " << m_cardType << endl
      << "Wrong number of electrons. Tried to set " << nonIsoEm.size() << " electrons" << endl;
    }
  }
}

/// set the Mip bits
void L1GctSourceCard::setMipBits(unsigned mip) {
  if (m_cardType!=cardType1) {
    throw cms::Exception("L1GctSetupError")
      << "L1GctSourceCard::setMipBits() : Source Card ID: " << m_id << " is of cardType " << m_cardType << endl
      << "and only handles regions" << endl;    
  }
  else {
    m_mipBits = mip & ((1<<N_MIP_BITS)-1);
  }
}

/// set the Quiet bits
void L1GctSourceCard::setQuietBits(unsigned quiet) {
  if (m_cardType!=cardType1) {
    throw cms::Exception("L1GctSetupError")
      << "L1GctSourceCard::setQuietBits() : Source Card ID: " << m_id << " is of cardType " << m_cardType << endl
      << "and only handles regions" << endl;
  }
  else {
    m_quietBits = quiet & ((1<<N_QUIET_BITS)-1);
  }
}

vector<L1CaloEmCand> L1GctSourceCard::getIsoElectrons() const
{
  if(m_cardType != cardType1)
  {
    throw cms::Exception("L1GctSetupError")
    << "L1GctSourceCard::getIsoElectrons() : Source Card ID: " << m_id << " is of cardType" << m_cardType
    << ". This Source Card is being asked for data handled only by cardType1 cards.\n";
  }
  return m_isoElectrons;
}

vector<L1CaloEmCand> L1GctSourceCard::getNonIsoElectrons() const
{
  if(m_cardType != cardType1)
  {
    throw cms::Exception("L1GctSetupError")
    << "L1GctSourceCard::getNonIsoElectrons() : Source Card ID: " << m_id << " is of cardType" << m_cardType
    << ". This Source Card is being asked for data handled only by cardType1 cards.\n";
  }
  return m_nonIsoElectrons;
}

unsigned L1GctSourceCard::getMipBits() const
{
  if(m_cardType != cardType1)
  {
    throw cms::Exception("L1GctSetupError")
    << "L1GctSourceCard::getMipBits() : Source Card ID: " << m_id << " is of cardType" << m_cardType
    << ". This Source Card is being asked for data handled only by cardType1 cards.\n";
  }
  return m_mipBits;
}

unsigned L1GctSourceCard::getQuietBits() const
{
  if(m_cardType != cardType1)
  {
    throw cms::Exception("L1GctSetupError")
    << "L1GctSourceCard::getQuietBits() : Source Card ID: " << m_id << " is of cardType" << m_cardType
    << ". This Source Card is being asked for data handled only by cardType1 cards.\n";
  }
  return m_quietBits;
}

vector<L1CaloRegion> L1GctSourceCard::getRegions() const
{
  if(m_cardType == cardType1)
  {
    throw cms::Exception("L1GctSetupError")
    << "L1GctSourceCard::getRegions() : Source Card ID: " << m_id << " is of cardType" << m_cardType
    << ". This Source Card is being asked for data handled only by cardType2 and cardType3 cards.\n";
  }
  return m_regions;
}

//Sets vectors according to what cables are being read in by the class
void L1GctSourceCard::setVectorSizes()
{
  //switch statement to set up the storage buffer vectors
  switch (m_cardType)
  {
  case cardType1:
    m_isoElectrons.resize(NUM_ELEC);
    m_nonIsoElectrons.resize(NUM_ELEC);
    break;

  case cardType2:
    m_regions.resize(NUM_REG_TYPE2);
    for (unsigned i=0; i<m_regions.size(); ++i) {
      L1CaloRegion temp(m_id, i, 0, false, false, false, false);
      if ((temp.gctCard() == m_id) && (temp.gctRegionIndex() == i)) {
	m_regions.at(i) = temp;
      } else {
	throw cms::Exception("L1GctSetupError")
	  << "L1GctSourceCard::setVectorSizes() : In Source Card ID: " << m_id
	  << " initialisation of position info for offset " << i << " has failed!\n!";
      }
    }
    break;
        
  case cardType3:
    m_regions.resize(NUM_REG_TYPE3);
    for (unsigned i=0; i<m_regions.size(); ++i) {
      L1CaloRegion temp(m_id, i, 0, false, false, false, false);
      if ((temp.gctCard() == m_id) && (temp.gctRegionIndex() == i)) {
	m_regions.at(i) = temp;
      } else {
	throw cms::Exception("L1GctSetupError")
	  << "L1GctSourceCard::setVectorSizes() : In Source Card ID: " << m_id
	  << " initialisation of position info for offset " << i << " has failed!\n!";
      }
    }
    break;

  default:
    throw cms::Exception("L1GctSetupError")
    << "L1GctSourceCard::setVectorSizes() : In Source Card ID: " << m_id 
    << ", an invalid cardType (cardType" << m_cardType << ") is being used!\n";
  }    
}


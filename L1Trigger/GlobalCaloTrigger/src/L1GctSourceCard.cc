#include "L1Trigger/GlobalCaloTrigger/interface/L1GctSourceCard.h"

#include "FWCore/Utilities/interface/Exception.h"

#include <iostream>

using namespace std;

L1GctSourceCard::L1GctSourceCard(int id, SourceCardType typeVal):
  m_id(id),
  m_cardType(typeVal)
{
    this->setVectorSizes();
}

L1GctSourceCard::~L1GctSourceCard()
{
}

std::ostream& operator << (std::ostream& os, const L1GctSourceCard& card)
{
  os << "SC ID " << card.m_id;
  os << " Type " << card.m_cardType;
  os << " File handle " << card.m_fin;
  os << " BX " << card.m_currentBX << std::endl;
  os << "No. of IsoElec " << card.m_isoElectrons.size() << std::endl;
  for(uint i=0; i < card.m_isoElectrons.size(); i++)
    {
      os << card.m_isoElectrons[i];
    } 
  os << "No. of NonIsoElec " << card.m_nonIsoElectrons.size() << std::endl;
  for(uint i=0; i < card.m_nonIsoElectrons.size(); i++)
    {
      os << card.m_nonIsoElectrons[i];
    }
  os << "MIPS " << card.m_mipBits;
  os << " QUIET " << card.m_quietBits << std::endl;
  os << "No. of Regions " << card.m_regions.size() << std::endl;
  for(uint i=0; i < card.m_regions.size(); i++)
    {
      os << card.m_regions[i];
    }
  return os;
}

void L1GctSourceCard::openInputFile(string fileName)
{
  //Opens the file
  m_fin.open(fileName.c_str(), ios::in);

  if(!m_fin.good())
  {
    throw cms::Exception("L1GctFileReadError")
      << "L1GctSourceCard::openInputFile() : Source Card ID: " << m_id
      << " couldn't open the file " + fileName + " for reading!\n";
  }
}    
    
void L1GctSourceCard::readBX()
{
  //Selects the method to read the relevant cable data from file
  switch (m_cardType)
  {
  case cardType1:  //get info off cables 1 & 2
    this->getCables1And2();
    break;

  case cardType2:  //get info off cables 3 & 4
    this->getCables3And4();
    break;
        
  case cardType3:  //get info off cables 5 & 6
    this->getCables5And6();
    break;

  default:
    throw cms::Exception("L1GctSetupError")
    << "L1GctSourceCard::readBx() : In Source Card ID: " << m_id 
    << ", an invalid cardType (cardType" << m_cardType << ") is being used!\n";
  }
}

void L1GctSourceCard::reset()
{
  m_fin.close();
  m_fin.clear();
    
  m_isoElectrons.clear();
  m_nonIsoElectrons.clear();
  m_mipBits.reset();
  m_quietBits.reset();
  m_regions.clear();
    
  setVectorSizes();
   
  m_currentBX = 0;
}

void L1GctSourceCard::fetchInput() 
{
}

void L1GctSourceCard::process()
{
}

vector<L1GctEmCand> L1GctSourceCard::getIsoElectrons() const
{
  if(m_cardType != cardType1)
  {
    throw cms::Exception("L1GctSetupError")
    << "L1GctSourceCard::getIsoElectrons() : Source Card ID: " << m_id << " is of cardType" << m_cardType
    << ". This Source Card is being asked for data handled only by cardType1 cards.\n";
  }
  return m_isoElectrons;
}

vector<L1GctEmCand> L1GctSourceCard::getNonIsoElectrons() const
{
  if(m_cardType != cardType1)
  {
    throw cms::Exception("L1GctSetupError")
    << "L1GctSourceCard::getNonIsoElectrons() : Source Card ID: " << m_id << " is of cardType" << m_cardType
    << ". This Source Card is being asked for data handled only by cardType1 cards.\n";
  }
  return m_nonIsoElectrons;
}

L1GctSourceCard::MipBits L1GctSourceCard::getMipBits() const
{
  if(m_cardType != cardType1)
  {
    throw cms::Exception("L1GctSetupError")
    << "L1GctSourceCard::getMipBits() : Source Card ID: " << m_id << " is of cardType" << m_cardType
    << ". This Source Card is being asked for data handled only by cardType1 cards.\n";
  }
  return m_mipBits;
}

L1GctSourceCard::QuietBits L1GctSourceCard::getQuietBits() const
{
  if(m_cardType != cardType1)
  {
    throw cms::Exception("L1GctSetupError")
    << "L1GctSourceCard::getQuietBits() : Source Card ID: " << m_id << " is of cardType" << m_cardType
    << ". This Source Card is being asked for data handled only by cardType1 cards.\n";
  }
  return m_quietBits;
}

vector<L1GctRegion> L1GctSourceCard::getRegions() const
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
    break;
        
  case cardType3:
    m_regions.resize(NUM_REG_TYPE3);
    break;

  default:
    throw cms::Exception("L1GctSetupError")
    << "L1GctSourceCard::setVectorSizes() : In Source Card ID: " << m_id 
    << ", an invalid cardType (cardType" << m_cardType << ") is being used!\n";
  }    
}

void L1GctSourceCard::getCables1And2()
{
  assert(!m_fin.eof());
    
  this->readBxNum();

  int i;  //counter
    
  hex(m_fin); //want hexadecimal
    
  ULong uLongBuffer;
  //get our 2 sets of electrons
  for(i=0; i < NUM_ELEC; ++i)
  {
    m_fin >> uLongBuffer;
    m_isoElectrons[i] = L1GctEmCand(uLongBuffer);
  }
  for(i=0; i < NUM_ELEC; ++i)
  {
    m_fin >> uLongBuffer;
    m_nonIsoElectrons[i] = L1GctEmCand(uLongBuffer);
  }

  bool bitBuffer;        
  //get the mip and quiet bits
  for(i=0; i < MIP_BITWIDTH; ++i)
  {
    m_fin >> bitBuffer;
    m_mipBits[i] = bitBuffer;
  }
  for(i=0; i < QUIET_BITWIDTH; ++i)
  {
    m_fin >> bitBuffer;
    m_quietBits[i] = bitBuffer;
  }
    
  //Skip remaining data in the bx
  for(i=0; i < NUM_REG_TYPE2 + NUM_REG_TYPE3; ++i)
  {
    m_fin >> uLongBuffer;
  }
    
  return;
}

void L1GctSourceCard::getCables3And4()
{
  assert(!m_fin.eof());
       
  this->readBxNum();

  int i;  //counter
   
  hex(m_fin); //want hexadecimal

  ULong uLongBuffer;
            
  //Skip some data
  for(i=0; i < DATA_OFFSET_TYPE2; ++i)
  {
    m_fin >> uLongBuffer;
  }

  //get the four endcap/barrel regions for this cable
  for(i=0; i < 4; ++i)
  {
    m_fin >> uLongBuffer;
    m_regions[i] = L1GctRegion(uLongBuffer);            
  }

  //get the 8 forward regions
  for(i=4; i < NUM_REG_TYPE2; ++i)
  {
    m_fin >> uLongBuffer;
    m_regions[i].setEt(uLongBuffer);
  }

  return;
}

void L1GctSourceCard::getCables5And6()
{
  assert(!m_fin.eof()); 
       
  this->readBxNum();

  int i;  //counter
    
  hex(m_fin); //want hexadecimal

  ULong uLongBuffer;
            
  //Skip some data
  for(i=0; i < DATA_OFFSET_TYPE3; ++i)
  {
    m_fin >> uLongBuffer; 
  }
    
  //get the 10 endcap/barrel regions for this cable
  for(i=0; i < NUM_REG_TYPE3; ++i)
  {
    m_fin >> uLongBuffer;
    m_regions[i] = L1GctRegion(uLongBuffer);            
  }
    
  //Skip some more data
  for(i=0; i < NUM_REG_TYPE2; ++i)
  {
    m_fin >> uLongBuffer; 
  }    
  return;
}

// Reads the Bunch Crossing number from the file
void L1GctSourceCard::readBxNum()
{
  string tempWord;
    
  m_fin >> tempWord;
    
  assert(tempWord == "Crossing");
   
  dec(m_fin); //want decimal numbers.
  m_fin >> m_currentBX; 
    
  return;
}

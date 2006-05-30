#include "L1Trigger/GlobalCaloTrigger/interface/L1GctSourceCard.h"

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctRegion.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctEmCand.h"

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
  m_mipBits=0;
  m_quietBits=0;
  m_regions.clear();
    
  setVectorSizes();
   
  m_currentBX = 0;
}

void L1GctSourceCard::fetchInput() 
{
  this->readBX();
}


/// set the Regions
void L1GctSourceCard::setRegions(vector<L1GctRegion> regions) {
  if (m_cardType!=cardType2 || m_cardType!=cardType3) {
    throw cms::Exception("L1GctSetupError")
      << "L1GctSourceCard::setRegions() : Source Card ID: " << m_id << " is of cardType " << m_cardType << endl
      << "and only handles EM candidates" << endl;
  }
  else {
    if (m_cardType==cardType2 && regions.size()==12) {
    
    }
    else if (m_cardType==cardType3 && regions.size()==10) {

    }
    else {
    throw cms::Exception("L1GctSetupError")
      << "L1GctSourceCard::setRegions() : Source Card ID: " << m_id << " is of cardType " << m_cardType << endl
      << "Wrong number of regions. Tried to set " << regions.size() << " regions" << endl;
    }
  }
}

/// set the Iso Em candidates
void L1GctSourceCard::setIsoEm(vector<L1GctEmCand> isoEm) {
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
void L1GctSourceCard::setNonIsoEm(vector<L1GctEmCand> nonIsoEm) {
  if (m_cardType!=cardType1) {
    throw cms::Exception("L1GctSetupError")
      << "L1GctSourceCard::setNonIsoEm() : Source Card ID: " << m_id << " is of cardType " << m_cardType << endl
      << "and only handles regions" << endl;
  }
  else {
    if (nonIsoEm.size()==4) {
      m_isoElectrons = nonIsoEm;
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
  for(i=0; i < N_MIP_BITS; ++i)
  {
    m_fin >> bitBuffer;
    m_mipBits &= (bitBuffer & 0x1);
  }
  for(i=0; i < N_QUIET_BITS; ++i)
  {
    m_fin >> bitBuffer;
    m_quietBits &= (bitBuffer & 0x1);
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
    m_regions[i] = makeRegion(uLongBuffer);            
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
    m_regions[i] = makeRegion(uLongBuffer);            
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
    
  cout << tempWord << endl;
  assert(tempWord == "Crossing");
   
  dec(m_fin); //want decimal numbers.
  m_fin >> m_currentBX; 
    
  return;
}

// make region from file data
L1GctRegion L1GctSourceCard::makeRegion(ULong rctFileData) {
  int et = rctFileData & 0x3ff;  //will put the first 10 bits of rawData into the Et

  rctFileData >>= 10;  //shift the remaining bits down to remove the 10 bits of Et

  bool overFlow = (  (rctFileData & 0x1)       != 0); //LSB is now overflow bit
  bool tauVeto  = ( ((rctFileData & 0x2) >> 1) != 0); //2nd bit is tauveto

  return L1GctRegion(0, 0, et, false, false, tauVeto, overFlow);

}

// make EM cand from file data
L1GctEmCand L1GctSourceCard::makeEmCand(ULong rctFileData) {
    unsigned rank = rctFileData & 0x3f;
    rctFileData >>= 6;   //shift the remaining bits down, to remove the rank info         
    int phi = rctFileData & 0x1;  //1 bit of Phi
    int eta = (rctFileData & 0xE) >> 1;  //other 3 bits are eta

    return L1GctEmCand(rank, eta, phi);
}

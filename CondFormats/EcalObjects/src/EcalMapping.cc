#include "CondFormats/EcalObjects/interface/EcalMapping.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include <cmath>
#include <map>

/****************\
- Public Methods -
\****************/


EcalMapping::EcalMapping()
{
}

EcalMapping::~EcalMapping()
{
}

const EcalMapping::crystalIndexPair EcalMapping::crystalNumberToIndex(int crystalNumber) const
{
  EcalMapping::crystalIndexPair indices;
  indices.i = (int) floor((crystalNumber-1)/numCrystalsInJ);
  indices.j = (crystalNumber-1) - (numCrystalsInJ * indices.i);

  return indices;
}

const EcalMapping::crystalAnglesPair EcalMapping::crystalNumberToAngles(int SM, int crystalNumber) const
{
  EcalMapping::crystalIndexPair indices = EcalMapping::crystalNumberToIndex( crystalNumber);
  
  int zside = 1;
  if (SM > 18) {
    SM = SM - 18;
    zside = -1;
  }

  EcalMapping::crystalAnglesPair angles;
  angles.ieta = (indices.i + 1) * zside;
  angles.iphi = ((SM-1) * numCrystalsInJ) + (numCrystalsInJ - 1 - indices.j) + 1;

  return angles;
}

const cms::EBDetId EcalMapping::crystalNumberToEBDetID(int SM, int crystalNumber) const
{
  EcalMapping::crystalAnglesPair angles = EcalMapping::crystalNumberToAngles(SM, crystalNumber);
  return cms::EBDetId(angles.ieta, angles.iphi);
}

int EcalMapping::crystalNumberToLogicID(int SM, int crystalNumber) const
{
  char logicId[11];
  // EB_crystal_number prefix hard-coded here
  sprintf(logicId, "1011%02d%04d", SM, crystalNumber);
  
  return atoi(logicId);
}

void EcalMapping::buildMapping()
{

  int logicId;
  cms::EBDetId detid;
  for (int SM = 1; SM <= numSuperModules; ++SM) {
    for (int xtal = 1; xtal <= numCrystalsPerSM; ++xtal) {
      logicId = EcalMapping::crystalNumberToLogicID(SM, xtal);
      detid = EcalMapping::crystalNumberToEBDetID(SM, xtal);
      m_channelToDetIdMap[logicId] = detid;
    }
  }
}

const cms::EBDetId EcalMapping::lookup(int channelId) const
{
  std::map< int, cms::EBDetId >::const_iterator i = m_channelToDetIdMap.find(channelId);
  if(i == m_channelToDetIdMap.end()) {
    return cms::EBDetId();
  }
  return i->second;
}

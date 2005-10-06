#ifndef EcalMapping_H
#define EcalMapping_H

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include <map>

class EcalMapping {
 public:
  EcalMapping();
  virtual ~EcalMapping();

  struct crystalIndexPair {
    int i;
    int j;
  };

  struct crystalAnglesPair {
    int ieta;
    int iphi;
  };

  const crystalIndexPair crystalNumberToIndex(int crystalNumber) const;

  const crystalAnglesPair crystalNumberToAngles(int SM, int crystalNumber) const;

  const EBDetId crystalNumberToEBDetID(int SM, int crystalNumber) const;
  
  int crystalNumberToLogicID(int SM, int crystalNumber) const;

  void buildMapping();

  /**
   *  Lookup the DetId given the online Condition database's logical ID
   */
  const EBDetId lookup(int channelId) const;

 private:
  static const int numCrystalsInI = 85;
  static const int numCrystalsInJ = 20;
  static const int numSuperModules = 36;
  static const int numCrystalsPerSM = 1700;

  std::map<int, EBDetId> m_channelToDetIdMap;
};


#endif

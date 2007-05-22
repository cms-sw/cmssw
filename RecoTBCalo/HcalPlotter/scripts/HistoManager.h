#ifndef HistoManager_included
#define HistoManager_included 1

#include "TDirectory.h"
#include "TH1.h"
#include "MyHcalClasses.h"

class HistoManager {
public:
  enum HistType { ENERGY=0, TIME=1, PULSE=2, ADC=3, NUMHISTTYPES=4 };
  enum EventType { UNKNOWN=0, PEDESTAL=1, LED=2, LASER=3, BEAM=4, NUMEVTTYPES=5 };

  HistoManager(TDirectory* parent);

  std::vector<MyHcalDetId> getDetIdsForType(HistType ht, EventType et);
  TH1* GetAHistogram(const MyHcalDetId& id, HistType ht, EventType et);


 

 std::vector<MyElectronicsId> getElecIdsForType(HistType ht, EventType et);
  TH1* GetAHistogram(const MyElectronicsId& id, HistType ht, EventType et);


 

  static std::string nameForFlavor(HistType ht);
  static std::string nameForEvent(EventType et);
private:
  bool m_writeMode;
  TDirectory* pedHistDir;
  TDirectory* ledHistDir;
  TDirectory* laserHistDir;
  TDirectory* beamHistDir;
  TDirectory* otherHistDir;
};

#endif

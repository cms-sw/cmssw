#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

namespace ecaldqm {
  const EcalElectronicsMapping* electronicsMap(0);
  const EcalTrigTowerConstituentsMap* trigtowerMap(0);

  unsigned nomemarr[] = {kEEm09, kEEm01, kEEm04, kEEm05, kEEm06, kEEp09, kEEp01, kEEp04, kEEp05, kEEp06};
  const std::set<unsigned> dccNoMEM(nomemarr, nomemarr + 10);
}

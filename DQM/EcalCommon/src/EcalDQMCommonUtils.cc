#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

namespace ecaldqm {
  const EcalElectronicsMapping* electronicsMap(0);
  const EcalTrigTowerConstituentsMap* trigtowerMap(0);
  const CaloGeometry* geometry(0);

  unsigned memarr[] = {kEEm07, kEEm08, kEEm02, kEEm03,
                       kEBm01, kEBm02, kEBm03, kEBm04, kEBm05, kEBm06, kEBm07, kEBm08, kEBm09,
                       kEBm10, kEBm11, kEBm12, kEBm13, kEBm14, kEBm15, kEBm16, kEBm17, kEBm18,
                       kEBp01, kEBp02, kEBp03, kEBp04, kEBp05, kEBp06, kEBp07, kEBp08, kEBp09,
                       kEBp10, kEBp11, kEBp12, kEBp13, kEBp14, kEBp15, kEBp16, kEBp17, kEBp18,
                       kEEp07, kEEp08, kEEp02, kEEp03};
  const std::vector<unsigned> memDCC(memarr, memarr + 44);

  const double etaBound(1.479);
}

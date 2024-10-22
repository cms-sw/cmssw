#ifndef CalibCalorimetry_CaloMiscalibTools_CaloMiscalibMapEcal_h
#define CalibCalorimetry_CaloMiscalibTools_CaloMiscalibMapEcal_h
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/CaloMiscalibMap.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include <iostream>
#include <iomanip>
#include <map>
#include <vector>

class CaloMiscalibMapEcal : public CaloMiscalibMap {
public:
  CaloMiscalibMapEcal() {}

  void prefillMap() {
    for (int iEta = -EBDetId::MAX_IETA; iEta <= EBDetId::MAX_IETA; ++iEta) {
      if (iEta == 0)
        continue;
      for (int iPhi = EBDetId::MIN_IPHI; iPhi <= EBDetId::MAX_IPHI; ++iPhi) {
        EBDetId ebdetid(iEta, iPhi);
        map_.setValue(ebdetid.rawId(), 1.0);
      }
    }

    for (int iX = EEDetId::IX_MIN; iX <= EEDetId::IX_MAX; ++iX) {
      for (int iY = EEDetId::IY_MIN; iY <= EEDetId::IY_MAX; ++iY) {
        EEDetId eedetidpos(iX, iY, 1);
        map_.setValue(eedetidpos.rawId(), 1.0);
        EEDetId eedetidneg(iX, iY, -1);
        map_.setValue(eedetidneg.rawId(), 1.0);
      }
    }
  }

  void addCell(const DetId &cell, float scaling_factor) override { map_.setValue(cell.rawId(), scaling_factor); }

  void print() {
    int icount = 0;
    for (int iEta = -EBDetId::MAX_IETA; iEta <= EBDetId::MAX_IETA; ++iEta) {
      if (iEta == 0)
        continue;
      for (int iPhi = EBDetId::MIN_IPHI; iPhi <= EBDetId::MAX_IPHI; ++iPhi) {
        if (EBDetId::validDetId(iEta, iPhi)) {
          EBDetId ebdetid(iEta, iPhi);
          EcalIntercalibConstantMap::const_iterator icalit = map_.find(ebdetid.rawId());
          EcalIntercalibConstant icalconst;
          icalconst = (*icalit);

          icount++;
          if (icount % 230 == 0) {
            std::cout << "here is value for chan eta/phi " << iEta << "/" << iPhi << "=" << icalconst << std::endl;
          }
        }
      }
    }
    for (int iX = EEDetId::IX_MIN; iX <= EEDetId::IX_MAX; ++iX) {
      for (int iY = EEDetId::IY_MIN; iY <= EEDetId::IY_MAX; ++iY) {
        if (EEDetId::validDetId(iX, iY, 1)) {
          EEDetId eedetidpos(iX, iY, 1);
          EcalIntercalibConstantMap::const_iterator icalit = map_.find(eedetidpos.rawId());
          EcalIntercalibConstant icalconst;
          icalconst = (*icalit);

          EEDetId eedetidneg(iX, iY, -1);
          EcalIntercalibConstantMap::const_iterator icalit2 = map_.find(eedetidneg.rawId());
          EcalIntercalibConstant icalconst2;
          icalconst2 = (*icalit2);

          icount++;
          if (icount % 230 == 0) {
            std::cout << "here is value for chan x/y " << iX << "/" << iY << " pos side is =" << icalconst
                      << " and neg side is= " << icalconst2 << std::endl;
          }
        }
      }
    }
  }

  const EcalIntercalibConstants &get() { return map_; }

private:
  EcalIntercalibConstants map_;
  const CaloSubdetectorGeometry *geometry;
};

#endif

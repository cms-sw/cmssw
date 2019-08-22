#ifndef GEOMETRY_HCALTOWERALGO_CALOTOWERHARDCODEGEOMETRYLOADER_H
#define GEOMETRY_HCALTOWERALGO_CALOTOWERHARDCODEGEOMETRYLOADER_H 1

#include "Geometry/HcalTowerAlgo/interface/CaloTowerGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTowerTopology.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"
#include <memory>
#include <vector>

/** \class CaloTowerHardcodeGeometryLoader
  *  
  * \author J. Mans - Minnesota
  */
class CaloTowerHardcodeGeometryLoader {
public:
  std::unique_ptr<CaloSubdetectorGeometry> load(const CaloTowerTopology *limits,
                                                const HcalTopology *hcaltopo,
                                                const HcalDDDRecConstants *hcons);

private:
  void makeCell(uint32_t din, CaloSubdetectorGeometry *geom) const;
  const CaloTowerTopology *m_limits;
  const HcalTopology *m_hcaltopo;
  const HcalDDDRecConstants *m_hcons;
  std::vector<double> theHBHEEtaBounds, theHFEtaBounds;
};

#endif

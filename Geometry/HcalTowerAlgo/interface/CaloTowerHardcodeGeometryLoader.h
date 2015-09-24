#ifndef GEOMETRY_HCALTOWERALGO_CALOTOWERHARDCODEGEOMETRYLOADER_H
#define GEOMETRY_HCALTOWERALGO_CALOTOWERHARDCODEGEOMETRYLOADER_H 1

#include "Geometry/HcalTowerAlgo/interface/CaloTowerGeometry.h"
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
  std::auto_ptr<CaloSubdetectorGeometry> load(const HcalTopology *hcaltopo, const HcalDDDRecConstants* hcons);
private:
  void makeCell(int ieta, int iphi, CaloSubdetectorGeometry* geom) const;
  const HcalTopology *m_hcaltopo;
  const HcalDDDRecConstants *m_hcons;
  std::vector<double> theHBHEEtaBounds, theHFEtaBounds;


};

#endif

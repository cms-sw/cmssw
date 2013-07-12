#ifndef RECOCALOTOOLS_SELECTORS_CALODUALCONESELECTOR_H
#define RECOCALOTOOLS_SELECTORS_CALODUALCONESELECTOR_H 1

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "RecoCaloTools/MetaCollections/interface/CaloRecHitMetaCollectionV.h"
#include <memory>

/** \class CaloDualConeSelector
  *  
  * $Date: $
  * $Revision: $
  * \author J. Mans - Minnesota
  */
class CaloDualConeSelector {
public:
  CaloDualConeSelector(double dRmin, double dRmax, const CaloGeometry* geom);
  CaloDualConeSelector(double dRmin, double dRmax, const CaloGeometry* geom, DetId::Detector detector, int subdet=0);

  std::auto_ptr<CaloRecHitMetaCollectionV> select(double eta, double phi, const CaloRecHitMetaCollectionV& inputCollection);
  std::auto_ptr<CaloRecHitMetaCollectionV> select(const GlobalPoint& p, const CaloRecHitMetaCollectionV& inputCollection);
private:
  const CaloGeometry* geom_;
  double deltaRmin_,deltaRmax_;
  DetId::Detector detector_;
  int subdet_;
};

#endif

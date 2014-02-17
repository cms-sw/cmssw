#ifndef RECOCALOTOOLS_SELECTORS_CALOCONESELECTOR_H
#define RECOCALOTOOLS_SELECTORS_CALOCONESELECTOR_H 1

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "RecoCaloTools/MetaCollections/interface/CaloRecHitMetaCollectionV.h"
#include <memory>

/** \class CaloConeSelector
  *  
  * $Date: 2006/08/29 12:49:10 $
  * $Revision: 1.1 $
  * \author J. Mans - Minnesota
  */
class CaloConeSelector {
public:
  CaloConeSelector(double dR, const CaloGeometry* geom);
  CaloConeSelector(double dR, const CaloGeometry* geom, DetId::Detector detector, int subdet=0);

  std::auto_ptr<CaloRecHitMetaCollectionV> select(double eta, double phi, const CaloRecHitMetaCollectionV& inputCollection);
  std::auto_ptr<CaloRecHitMetaCollectionV> select(const GlobalPoint& p, const CaloRecHitMetaCollectionV& inputCollection);
private:
  const CaloGeometry* geom_;
  double deltaR_;
  DetId::Detector detector_;
  int subdet_;
};

#endif

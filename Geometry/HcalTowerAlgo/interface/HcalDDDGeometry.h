#ifndef Geometry_HcalTowerAlgo_HcalDDDGeometry_h
#define Geometry_HcalTowerAlgo_HcalDDDGeometry_h

#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/HcalCommonData/interface/HcalCellType.h"

#include <vector>

class HcalDDDGeometry : public CaloSubdetectorGeometry {

public:

  explicit HcalDDDGeometry();
  /// The HcalDDDGeometry will delete all its cell geometries at destruction time
  virtual ~HcalDDDGeometry();
  
  virtual const std::vector<DetId>& getValidDetIds( DetId::Detector det    = DetId::Detector ( 0 ) , 
						    int             subdet = 0   ) const;

  virtual DetId getClosestCell(const GlobalPoint& r) const ;

  int insertCell (std::vector<HcalCellType::HcalCellType> const & );

private:

  mutable std::vector<DetId> m_validIds ;

  std::vector<HcalCellType::HcalCellType> hcalCells_;
  mutable DetId::Detector                 lastReqDet_;
  mutable int                             lastReqSubdet_;

  double                                  twopi, deg;
  double                                  etaMax_, firstHFQuadRing_;
};


#endif


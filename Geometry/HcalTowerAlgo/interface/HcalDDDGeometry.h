#ifndef Geometry_HcalTowerAlgo_HcalDDDGeometry_h
#define Geometry_HcalTowerAlgo_HcalDDDGeometry_h

#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/HcalCommonData/interface/HcalCellType.h"

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/IdealObliquePrism.h"
#include "Geometry/CaloGeometry/interface/IdealZPrism.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "CondFormats/AlignmentRecord/interface/HcalAlignmentRcd.h"
#include "Geometry/Records/interface/HcalGeometryRecord.h"

#include <vector>

class HcalDDDGeometry : public CaloSubdetectorGeometry {

public:

      typedef std::vector<IdealObliquePrism> HBCellVec ;
      typedef std::vector<IdealObliquePrism> HECellVec ;
      typedef std::vector<IdealObliquePrism> HOCellVec ;
      typedef std::vector<IdealZPrism>       HFCellVec ;

  explicit HcalDDDGeometry(const HcalTopology& theTopo);
  /// The HcalDDDGeometry will delete all its cell geometries at destruction time
  virtual ~HcalDDDGeometry();
  
  virtual const std::vector<DetId>& getValidDetIds( DetId::Detector det    = DetId::Detector ( 0 ) , 
						    int             subdet = 0   ) const;

  virtual DetId getClosestCell(const GlobalPoint& r) const ;

  int insertCell (std::vector<HcalCellType> const & );

      virtual void newCell( const GlobalPoint& f1 ,
			    const GlobalPoint& f2 ,
			    const GlobalPoint& f3 ,
			    const CCGFloat*    parm,
			    const DetId&       detId     ) ;
					
   protected:

      virtual const CaloCellGeometry* cellGeomPtr( uint32_t index ) const ;

private:

  mutable std::vector<DetId> m_validIds ;

  std::vector<HcalCellType> hcalCells_;

  const HcalTopology& topo_;
  mutable DetId::Detector                 lastReqDet_;
  mutable int                             lastReqSubdet_;

  double                                  twopi, deg;
  double                                  etaMax_, firstHFQuadRing_;

      HBCellVec m_hbCellVec ;
      HECellVec m_heCellVec ;
      HOCellVec m_hoCellVec ;
      HFCellVec m_hfCellVec ;
};


#endif


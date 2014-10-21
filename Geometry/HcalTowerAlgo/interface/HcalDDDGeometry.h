#ifndef Geometry_HcalTowerAlgo_HcalDDDGeometry_h
#define Geometry_HcalTowerAlgo_HcalDDDGeometry_h

#include "Geometry/HcalCommonData/interface/HcalCellType.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/IdealObliquePrism.h"
#include "Geometry/CaloGeometry/interface/IdealZPrism.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "CondFormats/AlignmentRecord/interface/HcalAlignmentRcd.h"
#include "Geometry/Records/interface/HcalGeometryRecord.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"
#include <atomic>

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

  void fillDetIds() const ;

  std::vector<HcalCellType> hcalCells_;
  CMS_THREAD_GUARD(m_filledDetIds) mutable std::vector<DetId> m_hbIds ;
  CMS_THREAD_GUARD(m_filledDetIds) mutable std::vector<DetId> m_heIds ;
  CMS_THREAD_GUARD(m_filledDetIds) mutable std::vector<DetId> m_hoIds ;
  CMS_THREAD_GUARD(m_filledDetIds) mutable std::vector<DetId> m_hfIds ;
  CMS_THREAD_GUARD(m_filledDetIds) mutable std::vector<DetId> m_emptyIds ;

  const HcalTopology& topo_;
  double              etaMax_;

  HBCellVec m_hbCellVec ;
  HECellVec m_heCellVec ;
  HOCellVec m_hoCellVec ;
  HFCellVec m_hfCellVec ;
  mutable std::atomic<bool> m_filledDetIds;
};

#endif


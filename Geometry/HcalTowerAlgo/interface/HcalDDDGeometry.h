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

class HcalDDDGeometryLoader;

class HcalDDDGeometry : public CaloSubdetectorGeometry {

public:

  friend class HcalDDDGeometryLoader;

  typedef std::vector<IdealObliquePrism> HBCellVec ;
  typedef std::vector<IdealObliquePrism> HECellVec ;
  typedef std::vector<IdealObliquePrism> HOCellVec ;
  typedef std::vector<IdealZPrism>       HFCellVec ;

  explicit HcalDDDGeometry(const HcalTopology& theTopo);
  /// The HcalDDDGeometry will delete all its cell geometries at destruction time
  ~HcalDDDGeometry() override;
  
  const std::vector<DetId>& getValidDetIds( DetId::Detector det    = DetId::Detector ( 0 ) , 
						    int             subdet = 0   ) const override;

  DetId getClosestCell(const GlobalPoint& r) const override ;

  int insertCell (std::vector<HcalCellType> const & );

  void newCell( const GlobalPoint& f1 ,
			const GlobalPoint& f2 ,
			const GlobalPoint& f3 ,
			const CCGFloat*    parm,
			const DetId&       detId     ) override ;
					
protected:

  const CaloCellGeometry* cellGeomPtr( uint32_t index ) const override ;

private:

  void newCellImpl( const GlobalPoint& f1 ,
			const GlobalPoint& f2 ,
			const GlobalPoint& f3 ,
			const CCGFloat*    parm,
			const DetId&       detId     ) ;

  //can only be used by friend classes, to ensure sorting is done at the end					
  void newCellFast( const GlobalPoint& f1 ,
			const GlobalPoint& f2 ,
			const GlobalPoint& f3 ,
			const CCGFloat*    parm,
			const DetId&       detId     ) ;

  void increaseReserve(unsigned int extra);
  void sortValidIds();

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


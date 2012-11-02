#ifndef HcalGeometry_h
#define HcalGeometry_h

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/IdealObliquePrism.h"
#include "Geometry/CaloGeometry/interface/IdealZPrism.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "CondFormats/AlignmentRecord/interface/HcalAlignmentRcd.h"
#include "Geometry/Records/interface/HcalGeometryRecord.h"

class HcalGeometry : public CaloSubdetectorGeometry {

public:
  
  typedef std::vector<IdealObliquePrism> HBCellVec ;
  typedef std::vector<IdealObliquePrism> HECellVec ;
  typedef std::vector<IdealObliquePrism> HOCellVec ;
  typedef std::vector<IdealZPrism>       HFCellVec ;

  typedef CaloCellGeometry::CCGFloat CCGFloat ;
  typedef CaloCellGeometry::Pt3D     Pt3D     ;
  typedef CaloCellGeometry::Pt3DVec  Pt3DVec  ;

  typedef HcalAlignmentRcd   AlignmentRecord ;
  typedef HcalGeometryRecord AlignedRecord   ;
  typedef PHcalRcd           PGeometryRecord ;
  typedef HcalDetId          DetIdType       ;

    
  enum { k_NumberOfParametersPerShape = 5 } ;

  static std::string dbString() { return "PHcalRcd" ; }

  virtual unsigned int numberOfShapes() const { return theTopology.getNumberOfShapes() ; }
  virtual unsigned int numberOfParametersPerShape() const { return k_NumberOfParametersPerShape ; }

  explicit HcalGeometry(const HcalTopology& topology);

  /// The HcalGeometry will delete all its cell geometries at destruction time
  virtual ~HcalGeometry();
  
  virtual const std::vector<DetId>& getValidDetIds(DetId::Detector det    = DetId::Detector ( 0 ), 
						   int             subdet = 0 ) const;

  virtual DetId getClosestCell(const GlobalPoint& r) const ;
      
  virtual CaloSubdetectorGeometry::DetIdSet getCells( const GlobalPoint& r,
						      double             dR ) const ;


  static std::string producerTag() { return "HCAL" ; }
  
  static unsigned int numberOfBarrelAlignments() { return 36 ; }

  static unsigned int numberOfEndcapAlignments() { return 36 ; }

  static unsigned int numberOfOuterAlignments() { return 36 ; }

  static unsigned int numberOfForwardAlignments() { return 60 ; }

  static unsigned int numberOfAlignments() 
    { return ( numberOfBarrelAlignments() +
	       numberOfEndcapAlignments() +
	       numberOfOuterAlignments() +
	       numberOfForwardAlignments() ) ; }

  static unsigned int alignmentTransformIndexLocal( const DetId& id ) ;

  static unsigned int alignmentTransformIndexGlobal( const DetId& id ) ;

  void localCorners( Pt3DVec&        lc  ,
		     const CCGFloat* pv  , 
		     unsigned int    i   ,
		     Pt3D&           ref   ) ;
  
  virtual void newCell( const GlobalPoint& f1 ,
			const GlobalPoint& f2 ,
			const GlobalPoint& f3 ,
			const CCGFloat*    parm,
			const DetId&       detId     ) ;

  virtual const CaloCellGeometry* getGeometry( const DetId& id ) const {
      return cellGeomPtr( theTopology.detId2denseId( id ) ) ;
  }

protected:

  virtual const CaloCellGeometry* cellGeomPtr( uint32_t index ) const ;

private:

  void fillDetIds() const ;

  void init() ;

  /// helper methods for getClosestCell
  int etaRing(HcalSubdetector bc, double abseta) const;
  int phiBin(double phi, int etaring) const;


  const HcalTopology& theTopology;
  
  mutable std::vector<DetId> m_hbIds ;
  mutable std::vector<DetId> m_heIds ;
  mutable std::vector<DetId> m_hoIds ;
  mutable std::vector<DetId> m_hfIds ;
  mutable std::vector<DetId> m_emptyIds ;

  HBCellVec m_hbCellVec ;
  HECellVec m_heCellVec ;
  HOCellVec m_hoCellVec ;
  HFCellVec m_hfCellVec ;
};


#endif


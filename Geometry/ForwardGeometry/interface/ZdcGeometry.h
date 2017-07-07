#ifndef Geometry_ForwardGeometry_ZdcGeometry_h
#define Geometry_ForwardGeometry_ZdcGeometry_h

#include "CondFormats/AlignmentRecord/interface/ZDCAlignmentRcd.h"
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/ForwardGeometry/interface/IdealZDCTrapezoid.h"
#include "Geometry/ForwardGeometry/interface/ZdcTopology.h"
#include "Geometry/Records/interface/ZDCGeometryRecord.h"
#include "Geometry/Records/interface/PZdcRcd.h"

class ZdcGeometry : public CaloSubdetectorGeometry 
{
   public:

      typedef std::vector<IdealZDCTrapezoid> CellVec ;

      typedef CaloCellGeometry::CCGFloat CCGFloat ;
      typedef CaloCellGeometry::Pt3D     Pt3D     ;
      typedef CaloCellGeometry::Pt3DVec  Pt3DVec  ;
      typedef CaloCellGeometry::Tr3D     Tr3D     ;

      typedef ZDCAlignmentRcd   AlignmentRecord ;
      typedef ZDCGeometryRecord AlignedRecord   ;
      typedef PZdcRcd           PGeometryRecord ;
      typedef HcalZDCDetId      DetIdType       ;

      enum { k_NumberOfCellsForCorners = HcalZDCDetId::kSizeForDenseIndexing } ;

      enum { k_NumberOfShapes = 3 } ;

      enum { k_NumberOfParametersPerShape = 4 } ;

      static std::string dbString() { return "PZdcRcd" ; }

      unsigned int numberOfShapes() const override { return k_NumberOfShapes ; }
      unsigned int numberOfParametersPerShape() const override { return k_NumberOfParametersPerShape ; }

      ZdcGeometry() ;

      explicit ZdcGeometry(const ZdcTopology * topology);
      ~ZdcGeometry() override;
  
//      virtual DetId getClosestCell(const GlobalPoint& r) const ;

      static std::string producerTag() { return "ZDC" ; }

      static unsigned int numberOfAlignments() { return 2 ; }

      static unsigned int alignmentTransformIndexLocal( const DetId& id ) ;

      static unsigned int alignmentTransformIndexGlobal( const DetId& id ) ;

      static void localCorners( Pt3DVec&        lc  ,
				const CCGFloat* pv  , 
				unsigned int    i   ,
				Pt3D&           ref   ) ;

      void newCell( const GlobalPoint& f1 ,
			    const GlobalPoint& f2 ,
			    const GlobalPoint& f3 ,
			    const CCGFloat*    parm,
			    const DetId&       detId     ) override ;

   protected:

      const CaloCellGeometry* cellGeomPtr( uint32_t index ) const override ;
					
   private:

      const ZdcTopology * theTopology;
      mutable DetId::Detector lastReqDet_;
      mutable int lastReqSubdet_;
      bool m_ownsTopology ;

      CellVec m_cellVec ;
};


#endif // Geometry_ForwardGeometry_ZdcGeometry_h


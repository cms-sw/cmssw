#ifndef Geometry_ForwardGeometry_ZdcGeometry_h
#define Geometry_ForwardGeometry_ZDcGeometry_h

#include "CondFormats/AlignmentRecord/interface/ZDCAlignmentRcd.h"
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/ForwardGeometry/interface/ZdcTopology.h"
#include "Geometry/Records/interface/ZDCGeometryRecord.h"
#include "Geometry/Records/interface/PZdcRcd.h"

class ZdcGeometry : public CaloSubdetectorGeometry 
{
   public:

      typedef ZDCAlignmentRcd   AlignmentRecord ;
      typedef ZDCGeometryRecord AlignedRecord   ;
      typedef PZdcRcd           PGeometryRecord ;
      typedef HcalZDCDetId      DetIdType       ;

      enum { k_NumberOfCellsForCorners = HcalZDCDetId::kSizeForDenseIndexing } ;

      enum { k_NumberOfShapes = 126 } ;

      enum { k_NumberOfParametersPerShape = 5 } ;

      static std::string dbString() { return "PZdcRcd" ; }

      virtual unsigned int numberOfShapes() const { return k_NumberOfShapes ; }
      virtual unsigned int numberOfParametersPerShape() const { return k_NumberOfParametersPerShape ; }

      ZdcGeometry() ;

      explicit ZdcGeometry(const ZdcTopology * topology);
      virtual ~ZdcGeometry();
  
      virtual const std::vector<DetId>& getValidDetIds( 
	 DetId::Detector det    = DetId::Detector ( 0 ) ,
	 int             subdet = 0   ) const;

      virtual DetId getClosestCell(const GlobalPoint& r) const ;

      static std::string producerTag() { return "ZDC" ; }

      static unsigned int numberOfAlignments() { return 0 ; }

      static unsigned int alignmentTransformIndexLocal( const DetId& id ) ;

      static unsigned int alignmentTransformIndexGlobal( const DetId& id ) ;

      static std::vector<HepPoint3D> localCorners( const double* pv, 
						   unsigned int  i,
						   HepPoint3D&   ref ) ;

      static CaloCellGeometry* newCell( const GlobalPoint& f1 ,
					const GlobalPoint& f2 ,
					const GlobalPoint& f3 ,
					CaloCellGeometry::CornersMgr* mgr,
					const double*      parm,
					const DetId&       detId     ) ;
					
   private:

      const ZdcTopology * theTopology;
      mutable DetId::Detector lastReqDet_;
      mutable int lastReqSubdet_;
      mutable std::vector<DetId> m_validIds;
      bool m_ownsTopology ;
};


#endif


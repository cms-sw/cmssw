#ifndef Geometry_ForwardGeometry_CastorGeometry_h
#define Geometry_ForwardGeometry_CastorGeometry_h 1

#include "CondFormats/AlignmentRecord/interface/CastorAlignmentRcd.h"
#include "DataFormats/HcalDetId/interface/HcalCastorDetId.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/ForwardGeometry/interface/IdealCastorTrapezoid.h"
#include "Geometry/ForwardGeometry/interface/CastorTopology.h"
#include "Geometry/Records/interface/CastorGeometryRecord.h"
#include "Geometry/Records/interface/PCastorRcd.h"

#include <vector>

class CastorGeometry : public CaloSubdetectorGeometry 
{
   public:

      typedef std::vector<IdealCastorTrapezoid> CellVec ;

      typedef CaloCellGeometry::CCGFloat CCGFloat ;
      typedef CaloCellGeometry::Pt3D     Pt3D     ;
      typedef CaloCellGeometry::Pt3DVec  Pt3DVec  ;
      typedef CaloCellGeometry::Tr3D     Tr3D     ;

      typedef CastorAlignmentRcd   AlignmentRecord ;
      typedef CastorGeometryRecord AlignedRecord   ;
      typedef PCastorRcd           PGeometryRecord ;
      typedef HcalCastorDetId      DetIdType       ;

      enum { k_NumberOfCellsForCorners = HcalCastorDetId::kSizeForDenseIndexing } ;

      enum { k_NumberOfShapes = 4 } ;

      enum { k_NumberOfParametersPerShape = 6 } ;

      static std::string dbString() { return "PCastorRcd" ; }

      virtual unsigned int numberOfTransformParms() const { return 3 ; }

      virtual unsigned int numberOfShapes() const { return k_NumberOfShapes ; }
      virtual unsigned int numberOfParametersPerShape() const { return k_NumberOfParametersPerShape ; }

      CastorGeometry() ;

      explicit CastorGeometry(const CastorTopology * topology);
      virtual ~CastorGeometry();

      virtual DetId getClosestCell(const GlobalPoint& r) const ;

      static std::string producerTag() { return "CASTOR" ; }

      static unsigned int numberOfAlignments() { return 1 ; }

      static unsigned int alignmentTransformIndexLocal( const DetId& id ) ;

      static unsigned int alignmentTransformIndexGlobal( const DetId& id ) ;

      static void localCorners( Pt3DVec&        lc  ,
				const CCGFloat* pv , 
				unsigned int    i  ,
				Pt3D&           ref  ) ;

      virtual void newCell( const GlobalPoint& f1 ,
			    const GlobalPoint& f2 ,
			    const GlobalPoint& f3 ,
			    const CCGFloat*    parm,
			    const DetId&       detId     ) ;

   protected:

      virtual const CaloCellGeometry* cellGeomPtr( uint32_t index ) const ;


private:

      const CastorTopology * theTopology;
      mutable DetId::Detector lastReqDet_;
      mutable int lastReqSubdet_;
      bool m_ownsTopology ;

      CellVec m_cellVec ;
};


#endif

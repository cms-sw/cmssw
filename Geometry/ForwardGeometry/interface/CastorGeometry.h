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

      unsigned int numberOfTransformParms() const override { return 3 ; }

      unsigned int numberOfShapes() const override { return k_NumberOfShapes ; }
      unsigned int numberOfParametersPerShape() const override { return k_NumberOfParametersPerShape ; }

      CastorGeometry() ;

      explicit CastorGeometry(const CastorTopology * topology);
      ~CastorGeometry() override;

      DetId getClosestCell(const GlobalPoint& r) const override ;

      static std::string producerTag() { return "CASTOR" ; }

      static unsigned int numberOfAlignments() { return 1 ; }

      static unsigned int alignmentTransformIndexLocal( const DetId& id ) ;

      static unsigned int alignmentTransformIndexGlobal( const DetId& id ) ;

      static void localCorners( Pt3DVec&        lc  ,
				const CCGFloat* pv , 
				unsigned int    i  ,
				Pt3D&           ref  ) ;

      void newCell( const GlobalPoint& f1 ,
			    const GlobalPoint& f2 ,
			    const GlobalPoint& f3 ,
			    const CCGFloat*    parm,
			    const DetId&       detId     ) override ;

   protected:

      const CaloCellGeometry* cellGeomPtr( uint32_t index ) const override ;


private:

      const CastorTopology * theTopology;
      mutable DetId::Detector lastReqDet_;
      mutable int lastReqSubdet_;
      bool m_ownsTopology ;

      CellVec m_cellVec ;
};


#endif

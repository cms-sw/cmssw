#ifndef GEOMETRY_HCALTOWERALGO_CALOTOWERGEOMETRY_H
#define GEOMETRY_HCALTOWERALGO_CALOTOWERGEOMETRY_H 1

#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
//#include "CondFormats/AlignmentRecord/interface/CaloTowerAlignmentRcd.h"
#include "Geometry/Records/interface/CaloTowerGeometryRecord.h"

/** \class CaloTowerGeometry
  *  
  * Only DetId::Calo, subdet=1 DetIds are handled by this class.
  *
  * $Date: 2009/01/29 22:28:52 $
  * $Revision: 1.4 $
  * \author J. Mans - Minnesota
  */
class CaloTowerGeometry : public CaloSubdetectorGeometry 
{
   public:

      typedef CaloTowerAlignmentRcd    AlignmentRecord ;
      typedef CaloTowerGeometryRecord  AlignedRecord   ;
      typedef PCaloTowerRcd            PGeometryRecord ;
      typedef CaloTowerDetId           DetIdType       ;

      enum { k_NumberOfCellsForCorners = CaloTowerDetId::kSizeForDenseIndexing } ;

      enum { k_NumberOfShapes = 41 } ;

      enum { k_NumberOfParametersPerShape = 5 } ;

      static std::string dbString() { return "PCaloTowerRcd" ; }

      virtual unsigned int numberOfShapes() const { return k_NumberOfShapes ; }
      virtual unsigned int numberOfParametersPerShape() const { return k_NumberOfParametersPerShape ; }


      CaloTowerGeometry();
      virtual ~CaloTowerGeometry();  

      static std::string producerTag() { return "TOWER" ; }

      static unsigned int numberOfAlignments() { return 0 ; }

      static unsigned int alignmentTransformIndexLocal( const DetId& id ) ;

      static unsigned int alignmentTransformIndexGlobal( const DetId& id ) ;

      static std::vector<HepGeom::Point3D<double> > localCorners( const double* pv, 
						   unsigned int  i,
						   HepGeom::Point3D<double> &   ref ) ;

      static CaloCellGeometry* newCell( const GlobalPoint& f1 ,
					const GlobalPoint& f2 ,
					const GlobalPoint& f3 ,
					CaloCellGeometry::CornersMgr* mgr,
					const double*      parm,
					const DetId&       detId     ) ;
					
};

#endif

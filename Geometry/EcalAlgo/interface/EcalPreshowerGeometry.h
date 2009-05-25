#ifndef EcalPreshowerGeometry_h
#define EcalPreshowerGeometry_h

#include "Geometry/EcalCommonData/interface/EcalPreshowerNumberingScheme.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/EcalPreshowerGeometryRecord.h"
#include "CondFormats/AlignmentRecord/interface/ESAlignmentRcd.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "Geometry/Records/interface/PEcalPreshowerRcd.h"
#include <vector>

class EcalPreshowerGeometry : public CaloSubdetectorGeometry
{
   public:

      typedef IdealGeometryRecord         IdealRecord   ;
      typedef EcalPreshowerGeometryRecord AlignedRecord ;
      typedef ESAlignmentRcd              AlignmentRecord ;
      typedef PEcalPreshowerRcd           PGeometryRecord ;

      typedef EcalPreshowerNumberingScheme NumberingScheme ;
      typedef CaloSubdetectorGeometry::ParVec ParVec ;
      typedef CaloSubdetectorGeometry::ParVecVec ParVecVec ;
      typedef ESDetId DetIdType ;

      enum { k_NumberOfCellsForCorners = ESDetId::kSizeForDenseIndexing } ;

      enum { k_NumberOfShapes = 2 } ;

      enum { k_NumberOfParametersPerShape = 3 } ;

      static std::string dbString() { return "PEcalPreshowerRcd" ; }

      virtual unsigned int numberOfShapes() const { return k_NumberOfShapes ; }
      virtual unsigned int numberOfParametersPerShape() const { return k_NumberOfParametersPerShape ; }

      EcalPreshowerGeometry() ;
  
      /// The EcalPreshowerGeometry will delete all its cell geometries at destruction time
      virtual ~EcalPreshowerGeometry();

      int getNumberOfWafers() const { return _nnwafers ; }

      int getNumberOfStripsPerWafer() const { return _nnstrips ; }

      void setzPlanes( float z1, float z2 ) { _zplane[0] = z1 ; _zplane[1] = z2 ; }

      void setNumberOfWafers( int nnwafers ) { _nnwafers=nnwafers ; }

      void setNumberOfStripsPerWafer( int nnstrips ) { _nnstrips=nnstrips ; }

      // Get closest cell
      virtual DetId getClosestCell( const GlobalPoint& r ) const ;


      // Get closest cell in arbitrary plane (1 or 2)
      virtual DetId getClosestCellInPlane( const GlobalPoint& r     ,
					   int                plane   ) const ;


      virtual void initializeParms() ;
      virtual unsigned int numberOfTransformParms() const { return 3 ; }

      static std::string hitString() { return "EcalHitsES" ; }

      static std::string producerTag() { return "EcalPreshower" ; }

      static unsigned int numberOfAlignments() { return 1 ; }

      static unsigned int alignmentTransformIndexLocal( const DetId& id ) ;

      static unsigned int alignmentTransformIndexGlobal( const DetId& id ) ;

      static std::vector<HepGeom::Point3D<double> > localCorners( const double* pv,
						   unsigned int  i,
						   HepGeom::Point3D<double> &   ref ) ;

      static CaloCellGeometry* newCell( const GlobalPoint& f1 ,
					const GlobalPoint& f2 ,
					const GlobalPoint& f3 ,
					CaloCellGeometry::CornersMgr* mgr,
					const double*      parm ,
					const DetId&       detId   ) ;

   private:

      /// number of modules
      int _nnwafers;
  
      /// number of crystals per module
      int _nnstrips; 

      double _act_w,_waf_w,_pitch,_intra_lad_gap,_inter_lad_gap,_centre_gap;
      double _zplane[2];
      
};


#endif


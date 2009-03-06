#ifndef EcalPreshowerGeometry_h
#define EcalPreshowerGeometry_h

#include "Geometry/EcalCommonData/interface/EcalPreshowerNumberingScheme.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/EcalPreshowerGeometryRecord.h"
#include "CondFormats/AlignmentRecord/interface/ESAlignmentRcd.h"
#include <vector>

class EcalPreshowerGeometry : public CaloSubdetectorGeometry
{
   public:

      typedef IdealGeometryRecord         IdealRecord   ;
      typedef EcalPreshowerGeometryRecord AlignedRecord ;
      typedef ESAlignmentRcd              AlignmentRecord ;

      typedef EcalPreshowerNumberingScheme NumberingScheme ;
      typedef CaloSubdetectorGeometry::ParVec ParVec ;
      typedef CaloSubdetectorGeometry::ParVecVec ParVecVec ;

      enum CornersCount { k_NumberOfCellsForCorners = 137216 } ;

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

      static std::string hitString() { return "EcalHitsES" ; }

      static std::string producerName() { return "EcalPreshower" ; }

      static unsigned int numberOfAlignments() { return 1 ; }

   private:

      /// number of modules
      int _nnwafers;
  
      /// number of crystals per module
      int _nnstrips; 

      float _act_w,_waf_w,_pitch,_intra_lad_gap,_inter_lad_gap,_centre_gap;
      float _zplane[2];
      
};


#endif


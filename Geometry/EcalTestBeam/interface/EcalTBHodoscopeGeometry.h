#ifndef Geometry_EcalTestBeam_EcalTBHodoscopeGeometry_HH
#define Geometry_EcalTestBeam_EcalTBHodoscopeGeometry_HH

#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/PreshowerStrip.h"

#include <vector>

class EcalTBHodoscopeGeometry : public CaloSubdetectorGeometry
{

   public:

      typedef std::vector<PreshowerStrip> CellVec ;

      typedef CaloCellGeometry::CCGFloat CCGFloat ;
      typedef CaloCellGeometry::Pt3D     Pt3D     ;
      typedef CaloCellGeometry::Pt3DVec  Pt3DVec  ;

      EcalTBHodoscopeGeometry() ;
      ~EcalTBHodoscopeGeometry() override ;

      void newCell( const GlobalPoint& f1 ,
			    const GlobalPoint& f2 ,
			    const GlobalPoint& f3 ,
			    const CCGFloat*    parm ,
			    const DetId&       detId   ) override ;

      static float getFibreLp( int plane, int fibre ) ;
      
      static float getFibreRp( int plane, int fibre ) ;
  
      static std::vector<int> getFiredFibresInPlane( float xtr, int plane ) ;

      static int getNPlanes() ;
      
      static int getNFibres() ;

   protected:

      const CaloCellGeometry* cellGeomPtr( uint32_t index ) const override ;
      
   private:
      
      struct fibre_pos 
      {
	    float lp, rp ;
      };

      static const int nPlanes_=4;
      static const int nFibres_=64;
      static const fibre_pos fibrePos_[ nPlanes_ ][ nFibres_ ] ;

      CellVec m_cellVec ;
};

#endif

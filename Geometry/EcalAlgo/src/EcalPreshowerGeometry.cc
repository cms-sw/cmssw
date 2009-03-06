#include "Geometry/CaloGeometry/interface/PreshowerStrip.h"
#include "Geometry/CaloGeometry/interface/CaloGenericDetId.h"
#include "Geometry/EcalAlgo/interface/EcalPreshowerGeometry.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <iostream>

EcalPreshowerGeometry::EcalPreshowerGeometry() :
   _nnwafers( 0 ) ,
   _nnstrips( 0 )
{
  //PM 20060518 FOR THE MOMENT USING HARDCODED NUMBERS
  //TODO: TAKE THEM FOR XML GEOMETRY

//  _zplane[0]=303.16;
//  _zplane[1]=307.13;
  _zplane[0]=303.353;
  _zplane[1]=307.838;
  _pitch = 0.190625; //strip pitch
  _waf_w = 6.3; // wafer width
  _act_w = 6.1; //wafer active area
  //new geometry
  _intra_lad_gap = 0.04; // gap between wafers in same ladder
  _inter_lad_gap = 0.05;// additional gap between wafers in adj ladders
  _centre_gap = 0.05;  // gap at center
}


EcalPreshowerGeometry::~EcalPreshowerGeometry() {}

unsigned int
EcalPreshowerGeometry::alignmentTransformIndexLocal( const DetId& id )
{
   const CaloGenericDetId gid ( id ) ;

   assert( gid.isES() ) ;
   unsigned int index ( 0 ) ;

   return index ;
}

unsigned int
EcalPreshowerGeometry::alignmentTransformIndexGlobal( const DetId& id )
{
   return (unsigned int)DetId::Ecal ;
}


void 
EcalPreshowerGeometry::initializeParms() 
{
   typedef CaloSubdetectorGeometry::CellCont Cont ;
   unsigned int n1 ( 0 ) ;
   unsigned int n2 ( 0 ) ;
   double z1 ( 0 ) ;
   double z2 ( 0 ) ;
   const Cont& con ( cellGeometries() ) ;
   for( unsigned int i ( 0 ) ; i != con.size() ; ++i )
   {
      const ESDetId esid ( getValidDetIds()[i] ) ;
      if( 1 == esid.plane() )
      {
	 z1 += fabs( con[i]->getPosition().z() ) ;
	 ++n1 ;
      }
      if( 2 == esid.plane() )
      {
	 z2 += fabs( con[i]->getPosition().z() ) ;
	 ++n2 ;
      }
//      if( 0 == z1 && 1 == esid.plane() ) z1 = fabs( i->second->getPosition().z() ) ;
//      if( 0 == z2 && 2 == esid.plane() ) z2 = fabs( i->second->getPosition().z() ) ;
//      if( 0 != z1 && 0 != z2 ) break ;
   }
   assert( 0 != n1 && 0 != n2 ) ;
   z1 /= (1.*n1) ;
   z2 /= (1.*n2) ;
   assert( 0 != z1 && 0 != z2 ) ;
   setzPlanes( z1, z2 ) ;
}

// Get closest cell, etc...
DetId 
EcalPreshowerGeometry::getClosestCell( const GlobalPoint& point ) const
{
  return getClosestCellInPlane( point, 2 );
} 

DetId 
EcalPreshowerGeometry::getClosestCellInPlane( const GlobalPoint& point,
					      int                plane          ) const
{
   float x = point.x();
   float y = point.y();
   float z = point.z();

   if (z == 0.0) 
   { 
      return DetId(0);
   }

  // extrapolate to plane on this side
  float xe = x * fabs(_zplane[plane-1]/z) ;
  float ye = y * fabs(_zplane[plane-1]/z) ;

  float x0,y0;

  if (plane == 1) {
    x0=xe;
    y0=ye;
  }
  else{
    y0=xe;
    x0=ye;
  }

  //find row

  int imul = (y0 < 0.) ? +1 : -1 ; 
  float yr = -(y0 + imul*_centre_gap )/_act_w;
  int row = (yr < 0.) ? (19 + int(yr) ) : (20 + int(yr));
  row= 40 - row;

  if (row < 1 || row > 40 ) {
    return DetId(0);
  }
  //find col
  int col = 40 ;
  int nlad = (col < 20 ) ? (20-col)/2 :(19-col)/2 ;
  float edge =  (20-col) * (_waf_w + _intra_lad_gap)+ nlad * _inter_lad_gap;
  edge = -edge;
  while (x0 < edge && col > 0){
    col--;
    nlad = (col < 20 ) ? (20-col)/2 :(19-col)/2 ;
    edge = (20-col) * (_waf_w + _intra_lad_gap) +          // complete wafer
      nlad * _inter_lad_gap;    // extra gap 
    edge = -edge;
  }

  col++;
  

  if ( col < 1 || col > 40 || x0 < edge) { 
    return DetId(0);
  }

  //Find strip
  float stredge = edge + (_waf_w + _intra_lad_gap)/2. -    // + half a wafer
    _act_w/2.  ;                                   // - half active area
  
  int istr = int((x0-stredge)/_pitch) + 1 ;
  if (istr > 32) istr=32;
  if (istr <1) istr=1;
  //Find zside
  int zside = ( z > 0.) ? +1 : -1;
  
  const DetId esid ( ESDetId::validDetId( istr,
					  1 == plane ? col : row,
					  1 == plane ? row : col,
					  plane,
					  zside  ) ?
		     DetId( ESDetId(             istr,
						 1 == plane ? col : row,
						 1 == plane ? row : col,
						 plane,
						 zside  ) ) :
		     DetId(0) ) ;

  return ( present( esid ) && 
	   !esid.null()       ? esid : DetId(0) ) ;
}

std::vector<HepPoint3D> 
EcalPreshowerGeometry::localCorners( const double* pv,
				     unsigned int  i,
				     HepPoint3D&   ref )
{
   return ( PreshowerStrip::localCorners( pv, ref ) ) ;
}

CaloCellGeometry* 
EcalPreshowerGeometry::newCell( const GlobalPoint& f1 ,
				const GlobalPoint& f2 ,
				const GlobalPoint& f3 ,
				CaloCellGeometry::CornersMgr* mgr,
				const double*      parm ,
				const DetId&       detId    ) 
{
   return ( new PreshowerStrip( f1, mgr , parm ) ) ;
}

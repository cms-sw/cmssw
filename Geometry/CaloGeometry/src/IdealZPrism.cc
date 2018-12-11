#include "Geometry/CaloGeometry/interface/IdealZPrism.h"
#include <cmath>

typedef IdealZPrism::CCGFloat CCGFloat ;
typedef IdealZPrism::Pt3D     Pt3D     ;
typedef IdealZPrism::Pt3DVec  Pt3DVec  ;

IdealZPrism::IdealZPrism()
  : CaloCellGeometry()
{}

namespace {

  // magic numbers determined by ParticleFlow
  constexpr float  EMDepthCorrection  = 22.;
  constexpr float  HADDepthCorrection = 25.;

  GlobalPoint correct(GlobalPoint const & ori, IdealZPrism::DEPTH depth) {
    if (depth==IdealZPrism::None) return ori;
    float zcorr = depth==IdealZPrism::EM ?EMDepthCorrection :  HADDepthCorrection;
    if (ori.z()<0) zcorr = -zcorr;
    return ori + GlobalVector(0.,0.,zcorr);
  }
}

IdealZPrism::IdealZPrism( const IdealZPrism& idzp ) 
  : CaloCellGeometry( idzp )
{
  if (idzp.forPF()) m_geoForPF.reset(new IdealZPrism(*idzp.forPF()));
}

IdealZPrism& 
IdealZPrism::operator=( const IdealZPrism& idzp ) 
{
  if( &idzp != this ) {
     CaloCellGeometry::operator=( idzp ) ;
     if (idzp.forPF()) m_geoForPF.reset(new IdealZPrism(*idzp.forPF()));
  }
  return *this ;
}

IdealZPrism::IdealZPrism( const GlobalPoint& faceCenter , 
			  CornersMgr*  mgr              ,
			  const CCGFloat*    parm       ,
			  IdealZPrism::DEPTH depth)
  : CaloCellGeometry ( faceCenter, mgr, parm ),
    m_geoForPF(depth==None ? nullptr : new IdealZPrism(correct(faceCenter,depth), mgr, parm, None ))
{initSpan();}

IdealZPrism::~IdealZPrism() 
{}

CCGFloat 
IdealZPrism::dEta() const 
{
   return param()[IdealZPrism::k_dEta] ;
}

CCGFloat 
IdealZPrism::dPhi() const 
{ 
   return param()[IdealZPrism::k_dPhi] ;
}

CCGFloat 
IdealZPrism::dz()   const 
{ 
   return param()[IdealZPrism::k_dZ] ;
}

CCGFloat 
IdealZPrism::eta()  const 
{
   return param()[IdealZPrism::k_Eta] ; 
}

CCGFloat 
IdealZPrism::z()    const 
{ 
   return param()[IdealZPrism::k_Z] ;
}

void 
IdealZPrism::vocalCorners( Pt3DVec&        vec ,
			   const CCGFloat* pv  ,
			   Pt3D&           ref   ) const 
{ 
   localCorners( vec, pv, ref ) ; 
}

GlobalPoint 
IdealZPrism::etaPhiR( float eta ,
		      float phi ,
		      float rad   ) 
{
   return GlobalPoint( rad*cosf(  phi )/coshf( eta ) ,
		       rad*sinf(  phi )/coshf( eta ) ,
		       rad*tanhf( eta )             ) ;
}

GlobalPoint 
IdealZPrism::etaPhiPerp( float eta , 
			 float phi , 
			 float perp  ) 
{
   return GlobalPoint( perp*cosf(  phi ) , 
		       perp*sinf(  phi ) , 
		       perp*sinhf( eta ) );
}

GlobalPoint 
IdealZPrism::etaPhiZ( float eta , 
		      float phi ,
		      float z    ) 
{
   return GlobalPoint( z*cosf( phi )/sinhf( eta ) ,
		       z*sinf( phi )/sinhf( eta ) ,
		       z                            ) ;
}

void
IdealZPrism::localCorners( Pt3DVec&        lc  ,
			   const CCGFloat* pv  ,
			   Pt3D&           ref   )
{
   assert( 8 == lc.size() ) ;
   assert( nullptr != pv ) ;
   
   const CCGFloat dEta ( pv[IdealZPrism::k_dEta] ) ;
   const CCGFloat dPhi ( pv[IdealZPrism::k_dPhi] ) ;
   const CCGFloat dz   ( pv[IdealZPrism::k_dZ] ) ;
   const CCGFloat eta  ( pv[IdealZPrism::k_Eta] ) ;
   const CCGFloat z    ( pv[IdealZPrism::k_Z] ) ;
   
   std::vector<GlobalPoint> gc ( 8, GlobalPoint(0,0,0) ) ;
   
   const GlobalPoint p ( etaPhiZ( eta, 0, z ) ) ;
   
   const float z_near ( z ) ;
   const float z_far  ( z*( 1 - 2*dz/p.mag() ) ) ;
   gc[ 0 ] = etaPhiZ( eta + dEta , +dPhi , z_near ) ; // (+,+,near)
   gc[ 1 ] = etaPhiZ( eta + dEta , -dPhi , z_near ) ; // (+,-,near)
   gc[ 2 ] = etaPhiZ( eta - dEta , -dPhi , z_near ) ; // (-,-,near)
   gc[ 3 ] = etaPhiZ( eta - dEta , +dPhi , z_near ) ; // (-,+,far)
   gc[ 4 ] = GlobalPoint( gc[0].x(), gc[0].y(), z_far ); // (+,+,far)
   gc[ 5 ] = GlobalPoint( gc[1].x(), gc[1].y(), z_far ); // (+,-,far)
   gc[ 6 ] = GlobalPoint( gc[2].x(), gc[2].y(), z_far ); // (-,-,far)
   gc[ 7 ] = GlobalPoint( gc[3].x(), gc[3].y(), z_far ); // (-,+,far)	
   
   for( unsigned int i ( 0 ) ; i != 8 ; ++i )
   {
      lc[i] = Pt3D( gc[i].x(), gc[i].y(), gc[i].z() ) ;
   }
   
   ref   = 0.25*( lc[0] + lc[1] + lc[2] + lc[3] ) ;
}

void
IdealZPrism::initCorners(CaloCellGeometry::CornersVec& co)
{
   if( co.uninitialized() ) 
   {
      CornersVec& corners ( co ) ;
      
      const GlobalPoint p      ( getPosition() ) ;
      const CCGFloat    z_near ( p.z() ) ;
      const CCGFloat    z_far  ( z_near + 2*dz()*p.z()/fabs( p.z() ) ) ;
      const CCGFloat    eta    ( p.eta() ) ;
      const CCGFloat    phi    ( p.phi() ) ;
      
      corners[ 0 ] = etaPhiZ( eta + dEta(), phi + dPhi(), z_near ); // (+,+,near)
      corners[ 1 ] = etaPhiZ( eta + dEta(), phi - dPhi(), z_near ); // (+,-,near)
      corners[ 2 ] = etaPhiZ( eta - dEta(), phi - dPhi(), z_near ); // (-,-,near)
      corners[ 3 ] = etaPhiZ( eta - dEta(), phi + dPhi(), z_near ); // (-,+,near)
      corners[ 4 ] = GlobalPoint( corners[0].x(), corners[0].y(), z_far ); // (+,+,far)
      corners[ 5 ] = GlobalPoint( corners[1].x(), corners[1].y(), z_far ); // (+,-,far)
      corners[ 6 ] = GlobalPoint( corners[2].x(), corners[2].y(), z_far ); // (-,-,far)
      corners[ 7 ] = GlobalPoint( corners[3].x(), corners[3].y(), z_far ); // (-,+,far)	
   }
}

std::ostream& operator<<( std::ostream& s, const IdealZPrism& cell ) 
{
   s << "Center: " <<  cell.getPosition() << std::endl ;
   s << "dEta = " << cell.dEta() << ", dPhi = " << cell.dPhi() << ", dz = " << cell.dz() << std::endl ;
   return s;
}

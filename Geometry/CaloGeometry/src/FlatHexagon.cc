#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/CaloGeometry/interface/FlatHexagon.h"
#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"
#include <algorithm>
#include <iostream>

//#define EDM_ML_DEBUG

typedef FlatHexagon::CCGFloat CCGFloat ;
typedef FlatHexagon::Pt3D     Pt3D     ;
typedef FlatHexagon::Pt3DVec  Pt3DVec  ;
typedef FlatHexagon::Tr3D     Tr3D     ;

typedef HepGeom::Vector3D<CCGFloat> FVec3D   ;
typedef HepGeom::Plane3D<CCGFloat>  Plane3D ;

typedef HepGeom::Vector3D<double> DVec3D ;
typedef HepGeom::Plane3D<double>  DPlane3D ;
typedef HepGeom::Point3D<double>  DPt3D ;

//----------------------------------------------------------------------

FlatHexagon::FlatHexagon() : CaloCellGeometry(),  m_axis ( 0., 0., 0. ),
			     m_corOne ( 0., 0., 0. ), m_local (0., 0., 0.),
			     m_global ( 0., 0., 0. ) {
}

FlatHexagon::FlatHexagon( const FlatHexagon& tr ) : CaloCellGeometry( tr ) {
  *this = tr ; 
}

FlatHexagon& FlatHexagon::operator=( const FlatHexagon& tr ) {
  CaloCellGeometry::operator=( tr ) ;
  if ( this != &tr ) {
    m_axis   = tr.m_axis ;
    m_corOne = tr.m_corOne ; 
    m_local  = tr.m_local;
    m_global = tr.m_global;
    m_tr     = tr.m_tr;
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("CaloGeometry") << "FlatHexagon(Copy): Local " << m_local 
				   << " Global " << m_global  << " eta " 
				   << etaPos() << " phi " << phiPos() 
				   << " Translation " << m_tr.getTranslation() 
				   << " and rotation " << m_tr.getRotation();
#endif
  return *this ; 
}

FlatHexagon::FlatHexagon( CornersMgr*  cMgr ,
			  const GlobalPoint& fCtr ,
			  const GlobalPoint& bCtr ,
			  const GlobalPoint& cor1 ,
			  const CCGFloat*    parV ) :
  CaloCellGeometry ( fCtr, cMgr, parV ) ,
  m_axis           ( ( bCtr - fCtr ).unit() ) ,
  m_corOne         ( cor1.x(), cor1.y(), cor1.z() ),
  m_local          (0., 0., 0.) {
  getTransform(m_tr,nullptr);
  Pt3D glb = m_tr*m_local;
  m_global = GlobalPoint(glb.x(),glb.y(),glb.z());
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("CaloGeometry") << "FlatHexagon: Local " << m_local 
				   << " Global " << glb << " eta "
				   << etaPos() << " phi " << phiPos() 
				   << " Translation " << m_tr.getTranslation()
				   << " and rotation " << m_tr.getRotation();
#endif
} 

FlatHexagon::FlatHexagon( const CornersVec& corn ,
			  const CCGFloat*   par    ) :
  CaloCellGeometry ( corn, par  ) , 
  m_corOne         ( corn[0].x(), corn[0].y(), corn[0].z() ),
  m_local          (0., 0., 0.) {
  getTransform(m_tr,nullptr);
  m_axis   = makeAxis();
  Pt3D glb = m_tr*m_local;
  m_global = GlobalPoint(glb.x(),glb.y(),glb.z());
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("CaloGeometry") << "FlatHexagon: Local " << m_local 
				   << " Global " << glb << " eta "
				   << etaPos() << " phi " << phiPos() 
				   << " Translation " << m_tr.getTranslation()
				   << " and rotation " << m_tr.getRotation();
#endif
} 

FlatHexagon::FlatHexagon( const FlatHexagon& tr, const Pt3D & local ) : 
  CaloCellGeometry( tr ),  m_local(local) {
  *this = tr;
  Pt3D glb = m_tr*m_local;
  m_global = GlobalPoint(glb.x(),glb.y(),glb.z());
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("CaloGeometry") << "FlatHexagon: Local " << m_local 
				   << " Global " << glb << " eta "
				   << etaPos() << " phi " << phiPos() 
				   << " Translation " << m_tr.getTranslation()
				   << " and rotation " << m_tr.getRotation();
#endif
}

FlatHexagon::~FlatHexagon() {}

GlobalPoint FlatHexagon::getPosition(const Pt3D& local ) const {
  Pt3D glb = m_tr*local;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("CaloGeometry") << "FlatHexagon::Local " << local.x() << ":"
				   << local.y() << ":" << local.z() 
				   << " Global " << glb.x() << ":" << glb.y() 
				   << ":" << glb.z() << " TR " << m_tr.xx() 
				   << ":" << m_tr.xy() << ":" << m_tr.xz() 
				   << ":" << m_tr.yx() << ":" << m_tr.yy() 
				   << ":" << m_tr.yz() << ":" << m_tr.zx() 
				   << ":" << m_tr.zy() << ":" << m_tr.zz() 
				   << ":" << m_tr.dx() << ":" << m_tr.dy() 
				   << ":" << m_tr.dz();
#endif
  return GlobalPoint(glb.x(),glb.y(),glb.z());
}

Pt3D FlatHexagon::getLocal(const GlobalPoint& global) const {
  Pt3D local = m_tr.inverse()*Pt3D(global.x(),global.y(),global.z());
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("CaloGeometry") << "FlatHexagon::Global " << global.x() 
				   << ":" << global.y() << ":" << global.z()
				   << " Local " << local.x() << ":" 
				   << local.y() << ":" << local.z();
#endif
  return local;
}

CCGFloat FlatHexagon::getThetaAxis() const { 
  return m_axis.theta() ; 
} 

CCGFloat FlatHexagon::getPhiAxis() const {
  return m_axis.phi() ; 
} 

const GlobalVector& FlatHexagon::axis() const { 
  return m_axis ; 
}

void FlatHexagon::vocalCorners( Pt3DVec&        vec ,
				const CCGFloat* pv  ,
				Pt3D&           ref  ) const { 
  localCorners( vec, pv, ref ) ; 
}

void  FlatHexagon::createCorners( const std::vector<CCGFloat>&  pv ,
				  const Tr3D&                   tr ,
				  std::vector<GlobalPoint>&     co   ) {

  assert(  3 == pv.size() ) ;
  assert( ncorner_ == co.size() ) ;

  Pt3DVec        ko ( ncorner_, Pt3D(0,0,0) ) ;

  Pt3D    tmp ;
  Pt3DVec to ( ncorner_, Pt3D(0,0,0) ) ;
  localCorners( to, &pv.front(), tmp ) ;

  for( unsigned int i ( 0 ) ; i != ncorner_ ; ++i ) {
    ko[i] = tr * to[i] ; // apply transformation
    const Pt3D & p ( ko[i] ) ;
    co[ i ] = GlobalPoint( p.x(), p.y(), p.z() ) ;
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("CaloGeometry") << "Corner[" << i << "] = " << co[i];
#endif
  }
}

void FlatHexagon::localCorners( Pt3DVec&        lc  ,
				const CCGFloat* pv  ,
				Pt3D&           ref   ) {
   assert( nullptr != pv ) ;
   assert( ncorner_ == lc.size() ) ;

   const CCGFloat dz ( pv[FlatHexagon::k_dZ] ) ;
   const CCGFloat r  ( pv[FlatHexagon::k_r] ) ;
   const CCGFloat R  ( pv[FlatHexagon::k_R] ) ;

   lc[0] = Pt3D (   0 ,  - 2*R , -dz ); // (0,-,-)
   lc[1] = Pt3D ( - r ,  -   R , -dz ); // (-,-,-)
   lc[2] = Pt3D ( - r ,      R , -dz ); // (-,+,-)
   lc[3] = Pt3D (   0 ,    2*R , -dz ); // (0,+,-)
   lc[4] = Pt3D (   r ,      R , -dz ); // (+,+,-)
   lc[5] = Pt3D (   r ,  -   R , -dz ); // (+,-,-)
   lc[6] = Pt3D (   0 ,  - 2*R ,  dz ); // (0,-,+)
   lc[7] = Pt3D ( - r ,  -   R ,  dz ); // (-,-,+)
   lc[8] = Pt3D ( - r ,      R ,  dz ); // (-,+,+)
   lc[9] = Pt3D (   0 ,    2*R ,  dz ); // (0,+,+)
   lc[10]= Pt3D (   r ,      R ,  dz ); // (+,+,+)
   lc[11]= Pt3D (   r ,  -   R ,  dz ); // (+,-,+)

   ref   = oneBySix_ * ( lc[0] + lc[1] + lc[2] + lc[3] + lc[4] + lc[5] );
#ifdef EDM_ML_DEBUG
   edm::LogVerbatim("CaloGeometry") << "Ref " << ref << " Local Corners " 
				    << lc[0] << "|" << lc[1] << "|" << lc[2] 
				    << "|" << lc[3] << "|" << lc[4] << "|" 
				    << lc[5] << "|" << lc[6] << "|" << lc[7] 
				    << "|" << lc[8] << "|" << lc[9] << "|"
				    << lc[10] << "|" << lc[11];
#endif
}

void FlatHexagon::getTransform( Tr3D& tr, Pt3DVec* lptr ) const {
  const GlobalPoint& p ( CaloCellGeometry::getPosition() ) ;
  const Pt3D         gFront ( p.x(), p.y(), p.z() ) ;
  const DPt3D        dgFront ( p.x(), p.y(), p.z() ) ;

  Pt3D  lFront ;
  assert( nullptr != param() ) ;
  std::vector<Pt3D > lc( ncorner_, Pt3D(0,0,0) ) ;
  localCorners( lc, param(), lFront ) ;

  // figure out if reflction volume or not

  Pt3D  lBack  ( (lc[6]+lc[7]+lc[8]+lc[9]+lc[10]+lc[11]) / 6.0 ) ;

  const DPt3D dlFront ( lFront.x(), lFront.y(), lFront.z() ) ;
  const DPt3D dlBack  ( lBack.x() , lBack.y() , lBack.z()  ) ;
  const DPt3D dlOne   ( lc[0].x() , lc[0].y() , lc[0].z()  ) ;

  const FVec3D dgAxis  ( axis().x(), axis().y(), axis().z() ) ;

  const DPt3D dmOne   ( m_corOne.x(), m_corOne.y(), m_corOne.z() ) ;

  const DPt3D dgBack  ( dgFront + ( dlBack - dlFront ).mag()*dgAxis ) ;
  DPt3D dgOne ( dgFront + ( dlOne - dlFront ).mag()*( dmOne - dgFront ).unit() ) ;

  const double dlangle ( ( dlBack - dlFront).angle( dlOne - dlFront ) ) ;
  const double dgangle ( ( dgBack - dgFront).angle( dgOne - dgFront ) ) ;
  const double dangle  ( dlangle - dgangle ) ;

  if( 1.e-6 < fabs(dangle) ) {//guard against precision problems
    const DPlane3D dgPl ( dgFront, dgOne, dgBack ) ;
    const DPt3D    dp2  ( dgFront + dgPl.normal().unit() ) ;

    DPt3D dgOld ( dgOne ) ;

    dgOne = ( dgFront + HepGeom::Rotate3D( -dangle, dgFront, dp2 )*
	      DVec3D( dgOld - dgFront ) ) ;
  }

  tr = Tr3D( dlFront , dlBack , dlOne ,
	     dgFront , dgBack , dgOne    ) ;

  if( nullptr != lptr ) (*lptr) = lc ;
}

void FlatHexagon::initCorners(CaloCellGeometry::CornersVec& co) {

  if( co.uninitialized() ) {
    CornersVec& corners ( co ) ;
    Pt3DVec lc ;
    Tr3D tr ;
    getTransform( tr, &lc ) ;

    for (unsigned int i ( 0 ) ; i != ncorner_ ; ++i ) {
      const Pt3D corn ( tr*lc[i] ) ;
      corners[i] = GlobalPoint( corn.x(), corn.y(), corn.z() ) ;
    }
  }
}

GlobalVector FlatHexagon::makeAxis() { 
  return GlobalVector( backCtr() - getPosition() ).unit() ;
}

GlobalPoint FlatHexagon::backCtr() const {
  float dz = (getCorners()[ncornerBy2_].z() > getCorners()[0].z()) ? 
    param()[FlatHexagon::k_dZ] : -param()[FlatHexagon::k_dZ];
  Pt3D local_b(m_local.x(),m_local.y(),m_local.z()+dz);
  Pt3D global_b = m_tr*local_b;
  GlobalPoint global(global_b.x(),global_b.y(),global_b.z());
  return global;
}

//----------------------------------------------------------------------
//----------------------------------------------------------------------

std::ostream& operator<<( std::ostream& s, const FlatHexagon& cell ) {
  s << "Center: " << cell.getPosition() << " eta " << cell.etaPos()
    << " phi " << cell.phiPos() << std::endl;
  s << "Axis: " << cell.getThetaAxis() << " " << cell.getPhiAxis() <<std::endl;
  const CaloCellGeometry::CornersVec& corners ( cell.getCorners() ) ;
  for ( unsigned int i=0 ; i != corners.size() ; ++i ) {
    s << "Corner: " << corners[i] << std::endl;
  }
  return s ;
}
  

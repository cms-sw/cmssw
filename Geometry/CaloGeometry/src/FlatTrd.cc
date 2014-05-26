#include "Geometry/CaloGeometry/interface/FlatTrd.h"
#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"
#include <algorithm>
#include <iostream>


typedef FlatTrd::CCGFloat CCGFloat ;
typedef FlatTrd::Pt3D     Pt3D     ;
typedef FlatTrd::Pt3DVec  Pt3DVec  ;
typedef FlatTrd::Tr3D     Tr3D     ;

typedef HepGeom::Vector3D<CCGFloat> FVec3D   ;
typedef HepGeom::Plane3D<CCGFloat>  Plane3D ;

typedef HepGeom::Vector3D<double> DVec3D ;
typedef HepGeom::Plane3D<double>  DPlane3D ;
typedef HepGeom::Point3D<double>  DPt3D ;

//----------------------------------------------------------------------

FlatTrd::FlatTrd() : CaloCellGeometry(),  m_axis ( 0., 0., 0. ),
		     m_corOne ( 0., 0., 0. ), m_local (0., 0., 0.),
		     m_global ( 0., 0., 0. ) {
}

FlatTrd::FlatTrd( const FlatTrd& tr ) : CaloCellGeometry( tr ) {
  *this = tr ; 
}

FlatTrd& FlatTrd::operator=( const FlatTrd& tr ) {
  CaloCellGeometry::operator=( tr ) ;
  if ( this != &tr ) {
    m_axis   = tr.m_axis ;
    m_corOne = tr.m_corOne ; 
    m_local  = tr.m_local;
    m_tr     = tr.m_tr;
    m_global = tr.m_global;
  }
  return *this ; 
}

FlatTrd::FlatTrd( const CornersMgr*  cMgr ,
		  const GlobalPoint& fCtr ,
		  const GlobalPoint& bCtr ,
		  const GlobalPoint& cor1 ,
		  const CCGFloat*    parV ) :
  CaloCellGeometry ( fCtr, cMgr, parV ) ,
  m_axis           ( ( bCtr - fCtr ).unit() ) ,
  m_corOne         ( cor1.x(), cor1.y(), cor1.z() ),
  m_local          (0., 0., 0.) {
  getTransform(m_tr,0);
  Pt3D glb = m_tr*m_local;
  m_global = GlobalPoint(glb.x(),glb.y(),glb.z());
} 

FlatTrd::FlatTrd( const CornersVec& corn ,
		  const CCGFloat*   par    ) :
  CaloCellGeometry ( corn, par  ) , 
  m_corOne         ( corn[0].x(), corn[0].y(), corn[0].z() ),
  m_local          (0., 0., 0.) {
  getTransform(m_tr,0);
  m_axis   = makeAxis();
  Pt3D glb = m_tr*m_local;
  m_global = GlobalPoint(glb.x(),glb.y(),glb.z());
} 

FlatTrd::~FlatTrd() {}

void FlatTrd::setLocal(const GlobalPoint & local) {
  m_local  = Pt3D(local.x(),local.y(),local.z());
  Pt3D glb = m_tr*m_local;
  m_global = GlobalPoint(glb.x(),glb.y(),glb.z());
}

const CaloCellGeometry::CornersVec& FlatTrd::getCorners() const {
  const CornersVec& co ( CaloCellGeometry::getCorners() ) ;
  if( co.uninitialized() ) {
    CornersVec& corners ( setCorners() ) ;
    
    Pt3DVec lc ;

    Tr3D tr ;
    getTransform( tr, &lc ) ;

    for (unsigned int i ( 0 ) ; i != 8 ; ++i ) {
      const Pt3D corn ( tr*lc[i] ) ;
      corners[i] = GlobalPoint( corn.x(), corn.y(), corn.z() ) ;
    }
  }
  return co ;
}

CCGFloat FlatTrd::getThetaAxis() const { 
  return m_axis.theta() ; 
} 

CCGFloat FlatTrd::getPhiAxis() const {
  return m_axis.phi() ; 
} 

const GlobalVector& FlatTrd::axis() const { 
  return m_axis ; 
}

void FlatTrd::vocalCorners( Pt3DVec&        vec ,
			    const CCGFloat* pv  ,
			    Pt3D&           ref  ) const { 
  TruncatedPyramid::localCorners( vec, pv, ref ) ; 
}

void FlatTrd::getTransform( Tr3D& tr, Pt3DVec* lptr ) const {
  const GlobalPoint& p ( CaloCellGeometry::getPosition() ) ;
  const Pt3D         gFront ( p.x(), p.y(), p.z() ) ;
  const DPt3D        dgFront ( p.x(), p.y(), p.z() ) ;

  Pt3D  lFront ;
  assert( 0 != param() ) ;
  std::vector<Pt3D > lc( 8, Pt3D(0,0,0) ) ;
  TruncatedPyramid::localCorners( lc, param(), lFront ) ;

  // figure out if reflction volume or not

  Pt3D  lBack  ( 0.25*(lc[4]+lc[5]+lc[6]+lc[7]) ) ;

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

  if( 0 != lptr ) (*lptr) = lc ;
}

GlobalVector FlatTrd::makeAxis() { 
  return GlobalVector( backCtr() - getPosition() ).unit() ;
}

const GlobalPoint FlatTrd::backCtr() const {
  float dz = (getCorners()[4].z() > getCorners()[0].z()) ? 
    param()[0] : -param()[0];
  Pt3D local_b(m_local.x(),m_local.y(),m_local.z()+dz);
  Pt3D global_b = m_tr*local_b;
  GlobalPoint global(global_b.x(),global_b.y(),global_b.z());
  return global;
}

//----------------------------------------------------------------------
//----------------------------------------------------------------------

std::ostream& operator<<( std::ostream& s, const FlatTrd& cell ) {
  s << "Center: " << cell.getPosition() << std::endl;
  s << "Axis: " << cell.getThetaAxis() << " " << cell.getPhiAxis() <<std::endl;
  const CaloCellGeometry::CornersVec& corners ( cell.getCorners() ) ;
  for ( unsigned int i=0 ; i != corners.size() ; ++i ) {
    s << "Corner: " << corners[i] << std::endl;
  }
  return s ;
}
  

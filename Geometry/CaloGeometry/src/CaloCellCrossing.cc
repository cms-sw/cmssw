#include "Geometry/CaloGeometry/interface/Line3D.h"
#include "Geometry/CaloGeometry/interface/CaloCellCrossing.h"

CaloCellCrossing::CaloCellCrossing( const GlobalPoint&              gp ,
				    const GlobalVector&             gv ,
				    const CaloCellCrossing::DetIds* di ,
				    const CaloSubdetectorGeometry*  sg ,
				    DetId::Detector                 det ,
				    int                             subdet,
				    double                          small,
				    bool                            onewayonly ) :
   m_gp ( gp ) ,
   m_gv ( gv ) 
{
   const double eps ( fabs( small ) > 1.e-15 ? fabs( small ) : 1.e-10 ) ;
   static unsigned int k_rlen ( 30 ) ;
   m_detId.reserve( k_rlen ) ;
   m_ctr  .reserve( k_rlen ) ;
   m_entr .reserve( k_rlen ) ;
   m_exit .reserve( k_rlen ) ;
//------------------------------------------------------------
   const HepLine3D line ( HepGeom::Point3D<double> (  gp.x(), gp.y(), gp.z() ), 
			  HepGeom::Vector3D<double> ( gv.x(), gv.y(), gv.z() ), eps ) ;

//   std::cout<<"*** Line: pt="<<line.pt()<<", unitvec="<<line.uv()<<std::endl ;

   const DetIds& ids ( 0 == di ? sg->getValidDetIds( det, subdet ) : *di ) ;
//------------------------------------------------------------
   for( DetIds::const_iterator id ( ids.begin() ) ; id != ids.end() ; ++id )
   {
      const DetId dId ( *id ) ;
      unsigned int found ( 0 ) ;
      const CaloCellGeometry& cg ( *sg->getGeometry( dId ) ) ;
      const CaloCellGeometry::CornersVec& gc ( cg.getCorners() ) ;
      const HepGeom::Point3D<double>  fr ( cg.getPosition().x(),
			    cg.getPosition().y(),
			    cg.getPosition().z()  ) ;
      const double bCut2 ( ( gc[0] - gc[6] ).mag2() ) ;

      if( ( !onewayonly ||
	    eps < HepGeom::Vector3D<double> ( fr - line.pt() ).dot( line.uv() ) ) &&
	  bCut2 > line.dist2( fr ) ) // first loose cut
      {
//	 std::cout<<"*** fr="<<fr<<", bCut ="<<sqrt(bCut2)<<", dis="<<line.dist(fr)<<std::endl ;
	 const HepGeom::Point3D<double>  cv[8] = 
	    { HepGeom::Point3D<double> ( gc[0].x(), gc[0].y(), gc[0].z() ) ,
	      HepGeom::Point3D<double> ( gc[1].x(), gc[1].y(), gc[1].z() ) ,
	      HepGeom::Point3D<double> ( gc[2].x(), gc[2].y(), gc[2].z() ) ,
	      HepGeom::Point3D<double> ( gc[3].x(), gc[3].y(), gc[3].z() ) ,
	      HepGeom::Point3D<double> ( gc[4].x(), gc[4].y(), gc[4].z() ) ,
	      HepGeom::Point3D<double> ( gc[5].x(), gc[5].y(), gc[5].z() ) ,
	      HepGeom::Point3D<double> ( gc[6].x(), gc[6].y(), gc[6].z() ) ,
	      HepGeom::Point3D<double> ( gc[7].x(), gc[7].y(), gc[7].z() ) } ;
	 const HepGeom::Point3D<double>  ctr ( 0.125*(cv[0]+cv[1]+cv[2]+cv[3]+
				       cv[4]+cv[5]+cv[6]+cv[7]) ) ;
	 const double dCut2 ( bCut2/4. ) ;
	 if( dCut2 > line.dist2( ctr ) ) // tighter cut
//	 if( 1 > line.dist2( ctr ) ) // tighter cut
	 {
//	    std::cout<<"** 2nd cut: ctr="<<ctr
//		     <<", dist="<<line.dist(ctr)<<std::endl ;
	    static const unsigned int nc[6][4] = 
	       { { 0,1,2,3 }, { 0,4,5,1 }, { 0,4,7,3 },
		 { 6,7,4,5 }, { 6,2,3,7 }, { 6,2,1,5 } } ;
	    for( unsigned int face ( 0 ) ; face != 6 ; ++face )
	    {
	       const unsigned int* ic ( &nc[face][0] ) ;
	       const HepGeom::Plane3D<double>  pl ( cv[ic[0]], cv[ic[1]], cv[ic[2]] ) ;
	       bool parallel ;
	       const HepGeom::Point3D<double>  pt ( line.point( pl, parallel ) ) ;
//	       std::cout<<"***Face: "<<face<<", pt="<<pt<<std::endl ;
	       if( !parallel )
	       {
//		  std::cout<<"Not parallel"<<std::endl ;
		  const HepLine3D la ( cv[ic[0]], cv[ic[1]], eps ) ;
		  const HepLine3D lb ( cv[ic[2]], cv[ic[3]], eps ) ;

//		  std::cout<<"la.point="<<la.point(pt)<<std::endl ;

//		  const double dot (  ( la.point( pt ) - pt ).dot( ( lb.point( pt ) - pt ) ) ) ;
//		  std::cout<<"***Dot1="<<dot<<std::endl;
		  if( eps > ( la.point( pt ) - pt ).dot( ( lb.point( pt ) - pt ) ) )
		  {
		     const HepLine3D lc ( cv[ic[0]], cv[ic[3]], eps ) ;
		     const HepLine3D ld ( cv[ic[1]], cv[ic[2]], eps ) ;
//		     const double dot (  ( lc.point( pt ) - pt ).dot( ( ld.point( pt ) - pt ) ) ) ;
//		     std::cout<<"***Dot2="<<dot<<std::endl;
		     if( eps > ( lc.point( pt ) - pt ).dot( ( ld.point( pt ) - pt ) ) )
		     {
			if( 0 == found )
			{
			   ++found ;
			   m_detId.push_back( dId ) ;
			   m_ctr .push_back( GlobalPoint( ctr.x(), ctr.y(), ctr.z() ) ) ;
			   m_entr.push_back( GlobalPoint(  pt.x(),  pt.y(),  pt.z() ) ) ;
			}
			else
			{
			   if( 1 == found )
			   {
			      ++found ;
			      m_exit.push_back( GlobalPoint( pt.x(), pt.y(), pt.z() ) ) ;
			   } 
			   else
			   {
			      const double dist1 (
				 ( pt - HepGeom::Point3D<double> ( m_entr.back().x(),
						    m_entr.back().y(),
						    m_entr.back().z() ) ).mag() ) ;
			      const double dist2 ( 
				 ( pt - HepGeom::Point3D<double> ( m_exit.back().x(),
						    m_exit.back().y(),
						    m_exit.back().z() ) ).mag() ) ;
			      if( eps < dist1 && 
				  eps < dist2    )
			      {
				 std::cout << "********For DetId = " << dId 
					   << " distances too big: "
					   << dist1 << ", " << dist2
					   << std::endl ;
			      }
			   }
			}
		     }
		  }
	       }
	    }
	 }
      }
      assert( 2 >= found ) ;
      if( 1 == found ) m_exit.push_back( m_entr.back() ) ;
   }
//------------------------------------------------------------
   assert( m_detId.size() == m_entr.size() &&
	   m_detId.size() == m_ctr .size() &&
	   m_detId.size() == m_exit.size()    ) ;

   m_len.reserve( m_entr.size() ) ;
   for( unsigned int i ( 0 ) ; i != m_entr.size() ; ++i )
   {
      m_len.push_back( ( m_exit[i] - m_entr[i] ).mag() ) ;
   }
}


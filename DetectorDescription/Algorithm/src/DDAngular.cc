#include "DetectorDescription/Algorithm/interface/DDAngular.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Base/interface/DDAlgoPar.h"
#include "DetectorDescription/Base/interface/DDutils.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

DDAngular::DDAngular( void )
  : m_n( 1 ),
    m_startCopyNo( 1 ),
    m_incrCopyNo( 1 ),
    m_startAngle( 0. ),
    m_rangeAngle( 360*deg ),
    m_radius( 0. ),
    m_delta( 0. )
{
  LogDebug( "DDAlgorithm" ) << "DDAngular: Creating an instance.";
}

DDAngular::~DDAngular( void ) 
{}

void
DDAngular::initialize( const DDNumericArguments & nArgs,
		       const DDVectorArguments & vArgs,
		       const DDMapArguments & ,
		       const DDStringArguments & sArgs,
		       const DDStringVectorArguments & )
{
  m_n           = int(nArgs["N"]);
  m_startCopyNo = int(nArgs["StartCopyNo"]);
  m_incrCopyNo  = int(nArgs["IncrCopyNo"]);
  m_rangeAngle  = nArgs["RangeAngle"];
  m_startAngle  = nArgs["StartAngle"];
  m_radius      = nArgs["Radius"];
  m_center      = vArgs["Center"];
  m_rotateSolid = vArgs["RotateSolid"];

  m_solidRot   = DDRotationMatrix();
  
  if( fabs( m_rangeAngle - 360.0 * CLHEP::deg ) < 0.001 * CLHEP::deg )
  { 
    m_delta = m_rangeAngle / double( m_n );
  }
  else
  {
    if( m_n > 1 )
    {
      m_delta = m_rangeAngle / double( m_n - 1 );
    }
    else
    {
      m_delta = 0.;
    }
  }  
  
  LogDebug( "DDAlgorithm" ) << "DDAngular: Parameters for position"
			    << "ing:: n " << m_n << " Start, Range, Delta " 
			    << m_startAngle/CLHEP::deg << " " 
			    << m_rangeAngle/CLHEP::deg << " " << m_delta/CLHEP::deg
			    << " Radius " << m_radius << " Centre " << m_center[0] 
			    << ", " << m_center[1] << ", " << m_center[2];
  
  //======= collect data concerning the rotation of the solid 
  typedef parE_type::mapped_type::size_type sz_type;
  sz_type sz = m_rotateSolid.size();
  if( sz%3 )
  {
    LogDebug( "DDAlgorithm" ) << "\trotateSolid must occur 3*n times (defining n subsequent rotations)\n"
			      << "\t  currently it appears " << sz << " times!\n";
  }
  for( sz_type i = 0; i < sz; i += 3 )
  {
    if(( m_rotateSolid[i] > 180. * deg ) || ( m_rotateSolid[i] < 0. ))
    {
      LogDebug( "DDAlgorithm" ) << "\trotateSolid \'theta\' must be in range [0,180*deg]\n"
				<< "\t  currently it is " << m_rotateSolid[i]/deg 
				<< "*deg in rotateSolid[" << double(i) << "]!\n";
    }
    DDAxisAngle temp( fUnitVector( m_rotateSolid[i], m_rotateSolid[i + 1] ),
		      m_rotateSolid[i + 2] );
    LogDebug( "DDAlgorithm" ) << "  rotsolid[" << i <<  "] axis=" << temp.Axis() << " rot.angle=" << temp.Angle()/deg;
    m_solidRot = temp * m_solidRot;			  
  }

  m_idNameSpace = DDCurrentNamespace::ns();
  m_childNmNs 	= DDSplit( sArgs["ChildName"] );
  if( m_childNmNs.second.empty())
    m_childNmNs.second = DDCurrentNamespace::ns();

  DDName parentName = parent().name();
  LogDebug( "DDAlgorithm" ) << "DDAngular: Parent " << parentName 
			    << "\tChild " << m_childNmNs.first << "\tNameSpace "
			    << m_childNmNs.second;
}

void
DDAngular::execute( DDCompactView& cpv )
{
  DDName mother = parent().name();
  DDName ddname( m_childNmNs.first, m_childNmNs.second );
  double theta  = 90.*CLHEP::deg;
  int    copy   = m_startCopyNo;
  double phi    = m_startAngle;

  for( int i = 0; i < m_n; ++i )
  {
    double phix = phi;
    double phiy = phix + 90. * CLHEP::deg;
    double phideg = phix / CLHEP::deg;

    std::string rotstr = m_childNmNs.first + "_" + dbl_to_string( phideg * 10.);
    DDRotation rotation = DDRotation( DDName( rotstr, m_idNameSpace ));
    if( !rotation )
    {
      LogDebug( "DDAlgorithm" ) << "DDAngular: Creating a new "
				<< "rotation: " << rotstr << "\t90., " 
				<< phix / CLHEP::deg << ", 90.," 
				<< phiy / CLHEP::deg << ", 0, 0";
	
      rotation = DDrot( DDName( rotstr, m_idNameSpace ), new DDRotationMatrix(( *DDcreateRotationMatrix( theta, phix, theta, phiy,
													 0., 0. ) * m_solidRot ))); 
    }
      
    double xpos = m_radius * cos( phi ) + m_center[0];
    double ypos = m_radius * sin( phi ) + m_center[1];
    double zpos = m_center[2];
    DDTranslation tran( xpos, ypos, zpos );
    
    cpv.position( ddname, mother, copy, tran, rotation );
    LogDebug( "DDAlgorithm" ) << "DDAngular: child " << m_childNmNs.second << ":" << m_childNmNs.first << " number " 
			      << copy << " positioned in " << mother << " at "
			      << tran << " with " << rotation << "\n";
    copy += m_incrCopyNo;
    phi  += m_delta;
  }
}

DD3Vector
DDAngular::fUnitVector( double theta, double phi )
{
  return DD3Vector( cos( phi ) * sin( theta ),
		    sin( phi ) * sin( theta ),
		    cos( theta ));
}

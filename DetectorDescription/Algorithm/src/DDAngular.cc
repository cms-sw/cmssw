#include "DetectorDescription/Algorithm/interface/DDAngular.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Base/interface/DDutils.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

DDAngular::DDAngular( void )
{
  LogDebug( "DDAlgorithm" ) << "DDAngular info: Creating an instance";
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
  n           = int(nArgs["N"]);
  startCopyNo = int(nArgs["StartCopyNo"]);
  incrCopyNo  = int(nArgs["IncrCopyNo"]);
  rangeAngle  = nArgs["RangeAngle"];
  startAngle  = nArgs["StartAngle"];
  radius      = nArgs["Radius"];
  center      = vArgs["Center"];
  rotateSolid = vArgs["RotateSolid"];

  solidRot_   = DDRotationMatrix();
  
  if( fabs( rangeAngle - 360.0 * CLHEP::deg ) < 0.001 * CLHEP::deg )
  { 
    delta = rangeAngle / double( n );
  }
  else
  {
    if( n > 1 )
    {
      delta = rangeAngle / double( n - 1 );
    }
    else
    {
      delta = 0.;
    }
  }  
  
  LogDebug( "DDAlgorithm" ) << "DDAngular debug: Parameters for position"
			    << "ing:: n " << n << " Start, Range, Delta " 
			    << startAngle/CLHEP::deg << " " 
			    << rangeAngle/CLHEP::deg << " " << delta/CLHEP::deg
			    << " Radius " << radius << " Centre " << center[0] 
			    << ", " << center[1] << ", "<<center[2];
  
  //======= collect data concerning the rotation of the solid 
  typedef parE_type::mapped_type::size_type sz_type;
  sz_type sz = rotateSolid.size();
  rotateSolid.clear();
  rotateSolid.resize( sz );
  if( sz%3 )
  {
    LogDebug( "DDAlgorithm" ) << "\trotateSolid must occur 3*n times (defining n subsequent rotations)\n"
			      << "\t  currently it appears " << sz << " times!\n";
  }
  for( sz_type i = 0; i < sz; i += 3 )
  {
    if(( rotateSolid[i] > 180. * deg ) || ( rotateSolid[i] < 0. ))
    {
      LogDebug( "DDAlgorithm" ) << "\trotateSolid \'theta\' must be in range [0,180*deg]\n"
				<< "\t  currently it is " << rotateSolid[i]/deg 
				<< "*deg in rotateSolid[" << double(i) << "]!\n";
    }
    
    DDAxisAngle temp( fUnitVector( rotateSolid[i], rotateSolid[i + 1] ),
		      rotateSolid[i + 2] );
    LogDebug( "DDAlgorithm" ) << "  rotsolid[" << i <<  "] axis=" << temp.Axis() << " rot.angle=" << temp.Angle()/deg;
    solidRot_ = temp * solidRot_;			  
  }
  
  idNameSpace = DDCurrentNamespace::ns();
  childName   = sArgs["ChildName"]; 
  
  DDName parentName = parent().name();
  LogDebug( "DDAlgorithm" ) << "DDAngular debug: Parent " << parentName 
			    << "\tChild " << childName << " NameSpace "
			    << idNameSpace;
}

void
DDAngular::execute( DDCompactView& cpv )
{
  DDName mother = parent().name();
  DDName child( DDSplit( childName ).first, DDSplit( childName ).second );
  double theta  = 90.*CLHEP::deg;
  int    copy   = startCopyNo;
  double phi    = startAngle;
  for( int i = 0; i < n; ++i )
  {
    double phix = phi;
    double phiy = phix + 90. * CLHEP::deg;
    double phideg = phix / CLHEP::deg;

    DDRotation rotation;
    DDRotationMatrix rotm1 = *DDcreateRotationMatrix( theta, phix, theta, phiy,
						      0., 0. ); // *rotation.rotation();
    DDRotationMatrix rotm2 = solidRot_;
    rotm2 = rotm1 * rotm2;

    if( phideg != 0 )
    {
      std::string rotstr = DDSplit( childName ).first + dbl_to_string( phideg * 10.);
      rotation = DDRotation( DDName( rotstr, idNameSpace ));
      if( !rotation )
      {
	LogDebug( "DDAlgorithm" ) << "DDAngular test: Creating a new "
				  << "rotation: " << rotstr << "\t90., " 
				  << phix / CLHEP::deg << ", 90.," 
				  << phiy / CLHEP::deg << ", 0, 0";
	
	rotation = DDrot( DDName( rotstr, idNameSpace), &rotm2 ); //theta, phix, theta, phiy,
	//0., 0. );
      }
    }
      
    double xpos = radius*cos( phi ) + center[0];
    double ypos = radius*sin( phi ) + center[1];
    double zpos = center[2];
    DDTranslation tran( xpos, ypos, zpos );
    
    cpv.position( child, mother, copy, tran, rotation );
    LogDebug( "DDAlgorithm" ) << "DDAngular test " << child << " number " 
			      << copy << " positioned in " << mother << " at "
			      << tran << " with " << rotation;
    copy += incrCopyNo;
    phi  += delta;
  }
}

DD3Vector
DDAngular::fUnitVector( double theta, double phi )
{
  return DD3Vector( cos( phi ) * sin( theta ),
		    sin( phi ) * sin( theta ),
		    cos( theta ));
}

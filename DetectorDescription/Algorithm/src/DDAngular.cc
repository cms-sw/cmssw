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
  
  if( rotationSolid.size()%3 )
    LogError( "DDAlgorithm" ) << "DDAngular error: rotateSolid must occur 3*n times (defining n subsequent rotations)\n";
  
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
  for (int i=0; i<n; i++) {
    double phix = phi;
    double phiy = phix + 90.*CLHEP::deg;
    double phideg = phix/CLHEP::deg;

    DDRotation rotation;
    if( phideg != 0 )
    {
      std::string rotstr = DDSplit( childName ).first+dbl_to_string( phideg*10.);
      rotation = DDRotation( DDName( rotstr, idNameSpace ));
      if( !rotation )
      {
	LogDebug( "DDAlgorithm" ) << "DDAngular test: Creating a new "
				  << "rotation: " << rotstr << "\t90., " 
				  << phix/CLHEP::deg << ", 90.," 
				  << phiy/CLHEP::deg <<", 0, 0";
	rotation = DDrot( DDName( rotstr, idNameSpace), theta, phix, theta, phiy,
			 0., 0.);
      }
    }
    
    double xpos = radius*cos(phi) + center[0];
    double ypos = radius*sin(phi) + center[1];
    double zpos = center[2];
    DDTranslation tran( xpos, ypos, zpos );
    
    cpv.position( child, mother, copy, tran, rotation );
    LogDebug( "DDAlgorithm" ) << "DDAngular test " << child << " number " 
			      << copy << " positioned in " << mother << " at "
			      << tran  << " with " << rotation;
    copy += incrCopyNo;
    phi  += delta;
  }
}

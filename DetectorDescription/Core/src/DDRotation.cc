#include <cstdio>
#include <atomic>
#include <cmath>
#include <sstream>
#include <string>

#include "DetectorDescription/Core/interface/DDRotationMatrix.h"
#include "DetectorDescription/Core/interface/DDTranslation.h"
#include "DetectorDescription/Core/interface/DDBase.h"
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDTransform.h"
#include "DetectorDescription/Core/interface/DDUnits.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Math/GenVector/AxisAngle.h"
#include "Math/GenVector/Cartesian3D.h"
#include "Math/GenVector/DisplacementVector3D.h"

using namespace dd::operators;

std::ostream & operator<<(std::ostream & os, const DDRotation & r)
{
  DDBase<DDName,DDRotationMatrix*>::def_type defined(r.isDefined());
  if (defined.first) {
    os << *(defined.first) << " ";
    if (defined.second) {
      const DDRotationMatrix & rm = r.rotation();
      DDAxisAngle   ra(rm);
      os << "t=" << CONVERT_TO( ra.Axis().Theta(), deg ) << "deg "
         << "p=" << CONVERT_TO( ra.Axis().Phi(), deg ) << "deg "
	 << "a=" << CONVERT_TO( ra.Angle(), deg ) << "deg"; 
    }
    else {
      os << "* rotation not defined * ";  
    }
  }  
  else {
    os << "* rotation not declared * ";  
  }  
  return os;
}

DDRotation::DDRotation()
  : DDBase< DDName, std::unique_ptr<DDRotationMatrix> >()
{
  constexpr char const* baseName = "DdBlNa";
  // In this particular case, we do not really care about multiple threads
  // using the same counter, we simply need to have a unique id for the 
  // blank matrix being created, so just making this static an atomic should do
  // the trick. In order to ensure repeatibility one should also include some 
  // some run specific Id, I guess. Not sure it really matters.
  static std::atomic<int> countBlank;
  char buf[64];
  snprintf( buf, 64, "%s%i", baseName, countBlank++ );
  create( DDName( buf, baseName ), std::make_unique<DDRotationMatrix>());
}

DDRotation::DDRotation( const DDName & name )
  : DDBase< DDName, std::unique_ptr<DDRotationMatrix>>()
{
  create( name );
}

DDRotation::DDRotation( const DDName & name, std::unique_ptr<DDRotationMatrix> rot )
  : DDBase< DDName, std::unique_ptr<DDRotationMatrix>>()
{
  create( name, std::move( rot ));
}

DDRotation::DDRotation( std::unique_ptr<DDRotationMatrix> rot )
  : DDBase< DDName, std::unique_ptr<DDRotationMatrix>>()
{
  static std::atomic<int> countNN;
  char buf[64];
  snprintf(buf, 64, "DdNoNa%i", countNN++);
  create( DDName( buf, "DdNoNa" ), std::move( rot ));
}

DDRotation
DDrot( const DDName & ddname, std::unique_ptr<DDRotationMatrix> rot )
{
   // memory of rot goes sto DDRotationImpl!!
  return DDRotation(ddname, std::move( rot ));
}

std::unique_ptr<DDRotation>
DDrotPtr( const DDName & ddname, std::unique_ptr<DDRotationMatrix> rot )
{
   // memory of rot goes sto DDRotationImpl!!
  return std::make_unique<DDRotation>( ddname, std::move( rot ));
}
 
// makes sure that the DDRotationMatrix constructed is right-handed and orthogonal.
DDRotation DDrot( const DDName & ddname,
		  double thetaX, double phiX,
		  double thetaY, double phiY,
		  double thetaZ, double phiZ)
{
   // define 3 unit std::vectors
  DD3Vector x(cos(phiX)*sin(thetaX), sin(phiX)*sin(thetaX), cos(thetaX));
  DD3Vector y(cos(phiY)*sin(thetaY), sin(phiY)*sin(thetaY), cos(thetaY));
  DD3Vector z(cos(phiZ)*sin(thetaZ), sin(phiZ)*sin(thetaZ), cos(thetaZ));
  
  double tol = 1.0e-3; // Geant4 compatible
  double check = (x.Cross(y)).Dot(z); // in case of a LEFT-handed orthogonal system this must be -1
  if (fabs(1.-check)>tol) {
    edm::LogError("DDRotation") << ddname << " is not a RIGHT-handed orthonormal matrix!" << std::endl;
    throw cms::Exception("DDException") << ddname.name() << " is not RIGHT-handed!";
  }
  
  return DDRotation( ddname,
		     std::make_unique<DDRotationMatrix>(x.x(),y.x(),z.x(),
							x.y(),y.y(),z.y(),
							x.z(),y.z(),z.z()));
}
  
DDRotation
DDrotReflect( const DDName & ddname, std::unique_ptr<DDRotationMatrix> rot )
{
  return DDRotation( ddname, std::move( rot ));
}

// makes sure that the DDRotationMatrix built is LEFT-handed coordinate system (i.e. reflected)
DDRotation DDrotReflect( const DDName & ddname,
			 double thetaX, double phiX,
			 double thetaY, double phiY,
			 double thetaZ, double phiZ )
{
  // define 3 unit std::vectors forming the new left-handed axes 
  DD3Vector x(cos(phiX)*sin(thetaX), sin(phiX)*sin(thetaX), cos(thetaX));
  DD3Vector y(cos(phiY)*sin(thetaY), sin(phiY)*sin(thetaY), cos(thetaY));
  DD3Vector z(cos(phiZ)*sin(thetaZ), sin(phiZ)*sin(thetaZ), cos(thetaZ));
  
  double tol = 1.0e-3; // Geant4 compatible
  double check = (x.Cross(y)).Dot(z); // in case of a LEFT-handed orthogonal system this must be -1
  if (fabs(1.+check)>tol) {
    edm::LogError("DDRotation") << ddname << " is not a LEFT-handed orthonormal matrix!" << std::endl;
    throw cms::Exception("DDException") << ddname.name() << " is not LEFT-handed!";
  }
  
  return DDRotation( ddname,
		     std::make_unique<DDRotationMatrix>(x.x(),y.x(),z.x(),
							x.y(),y.y(),z.y(),
							x.z(),y.z(),z.z()));  
}		

// does NOT check LEFT or Right handed coordinate system takes either.
std::unique_ptr<DDRotationMatrix>
DDcreateRotationMatrix( double thetaX, double phiX,
			double thetaY, double phiY,
			double thetaZ, double phiZ )
{
  // define 3 unit std::vectors forming the new left-handed axes 
  DD3Vector x(cos(phiX)*sin(thetaX), sin(phiX)*sin(thetaX), cos(thetaX));
  DD3Vector y(cos(phiY)*sin(thetaY), sin(phiY)*sin(thetaY), cos(thetaY));
  DD3Vector z(cos(phiZ)*sin(thetaZ), sin(phiZ)*sin(thetaZ), cos(thetaZ));
  
  double tol = 1.0e-3; // Geant4 compatible
  double check = (x.Cross(y)).Dot(z);// in case of a LEFT-handed orthogonal system this must be -1, RIGHT-handed: +1
  if ((1.-fabs(check))>tol) {
    std::ostringstream o;
    o << "matrix is not an (left or right handed) orthonormal matrix! (in deg)" << std::endl
      << " thetaX=" << CONVERT_TO( thetaX, deg ) << " phiX=" << CONVERT_TO( phiX, deg ) << std::endl
      << " thetaY=" << CONVERT_TO( thetaY, deg ) << " phiY=" << CONVERT_TO( phiY, deg ) << std::endl
      << " thetaZ=" << CONVERT_TO( thetaZ, deg ) << " phiZ=" << CONVERT_TO( phiZ, deg ) << std::endl;
    edm::LogError("DDRotation") << o.str() << std::endl;
     
    throw cms::Exception("DDException") << o.str();
  }
  
  return std::make_unique<DDRotationMatrix>(x.x(),y.x(),z.x(),
					    x.y(),y.y(),z.y(),
					    x.z(),y.z(),z.z());
}			 

DDRotation
DDanonymousRot( std::unique_ptr<DDRotationMatrix> rot )
{
  return DDRotation( std::move( rot ));
}

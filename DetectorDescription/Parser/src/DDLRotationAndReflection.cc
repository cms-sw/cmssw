/***************************************************************************
                          DDLRotationAndReflection.cc  -  description
                             -------------------
    begin                : Tue Aug 6 2002
    email                : case@ucdhep.ucdavis.edu
***************************************************************************/

/***************************************************************************
 *                                                                         *
 *           DDDParser sub-component of DDD                                *
 *                                                                         *
 ***************************************************************************/

#include "DetectorDescription/Parser/src/DDLRotationAndReflection.h"

#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDTransform.h"
#include "DetectorDescription/Base/interface/DDdebug.h"

#include "DetectorDescription/ExprAlgo/interface/ExprEvalSingleton.h"

//CLHEP dependency
#include "CLHEP/Units/GlobalSystemOfUnits.h"

#include <cmath>

DDLRotationAndReflection::DDLRotationAndReflection( DDLElementRegistry* myreg )
  : DDXMLElement( myreg ) 
{}

DDLRotationAndReflection::~DDLRotationAndReflection( void )
{}

void
DDLRotationAndReflection::processElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv )
{

  DCOUT_V('P', "DDLRotationAndReflection::processElement started " << name);

  DD3Vector x = makeX(nmspace);
  DD3Vector y = makeY(nmspace);
  DD3Vector z = makeZ(nmspace);

  DDXMLAttribute atts = getAttributeSet();


  if ((name == "Rotation") && isLeftHanded(x, y, z, nmspace) == 0)
  {
    DDRotationMatrix* ddr = new DDRotationMatrix(x, y, z);
    DDRotation ddrot = DDrot(getDDName(nmspace), ddr);
    DCOUT_V ('p', "Rotation created: " << ddrot << std::endl);
  }
  else if ((name == "Rotation")  && isLeftHanded(x, y, z, nmspace) == 1)
  {
    std::string msg("\nDDLRotationAndReflection attempted to make a");
    msg += " left-handed rotation with a Rotation element. If";
    msg += " you meant to make a reflection, use ReflectionRotation";
    msg += " elements, otherwise, please check your matrix.  Other";
    msg += " errors may follow.  Rotation  matrix not created.";
    edm::LogError("DetectorDescription_Parser_Rotation_and_Reflection") << msg << std::endl; // this could become a throwWarning or something.
  }
  else if (name == "ReflectionRotation" && isLeftHanded(x, y, z, nmspace) == 1) 
  {
    ExprEvalInterface & ev = ExprEvalSingleton::instance();
    DDRotation ddrot = 
      DDrotReflect(getDDName(nmspace)
		   , ev.eval(nmspace, atts.find("thetaX")->second)
		   , ev.eval(nmspace, atts.find("phiX")->second)
		   , ev.eval(nmspace, atts.find("thetaY")->second)
		   , ev.eval(nmspace, atts.find("phiY")->second)
		   , ev.eval(nmspace, atts.find("thetaZ")->second)
		   , ev.eval(nmspace, atts.find("phiZ")->second));
    DCOUT_V ('p', "Rotation created: " << ddrot << std::endl);
  }
  else if (name == "ReflectionRotation" && isLeftHanded(x, y, z, nmspace) == 0)
  {
    std::string msg("WARNING:  Attempted to make a right-handed");
    msg += " rotation using a ReflectionRotation element. ";
    msg += " If you meant to make a Rotation, use Rotation";
    msg += " elements, otherwise, please check your matrix.";
    msg += "  Other errors may follow.  ReflectionRotation";
    msg += " matrix not created.";
    edm::LogError("DetectorDescription_Parser_Rotation_and_Reflection") << msg << std::endl; // this could be a throwWarning or something.
  }
  else
  {
    std::string msg = "\nDDLRotationAndReflection::processElement tried to process wrong element.";
    throwError(msg);
  }
  // after a rotation or reflection rotation has been processed, clear it
  clear();

  DCOUT_V('P', "DDLRotationAndReflection::processElement completed");
}


// returns 1 if it is a left-handed CLHEP rotation matrix, 0 if not, but is okay, -1 if 
// it is not an orthonormal matrix.
//
// Upon encountering the end tag of a Rotation element, we've got to feed
// the appropriate rotation in to the DDCore.  This is an attempt to do so.
//
// Basically, I cannibalized code from g3tog4 (see http link below) and then
// provided the information from our DDL to the same calls.  Tim Cox showed me
// how to build the rotation matrix (mathematically) and the g3tog4 code basically
// did the rest.
//

int
DDLRotationAndReflection::isLeftHanded (DD3Vector x, DD3Vector y, DD3Vector z, const std::string & nmspace)
{
  DCOUT_V('P', "DDLRotation::isLeftHanded started");

  int ret = 0;

  /**************** copied and cannibalized code:
 
  from g3tog4
 
  http://atlassw1.phy.bnl.gov/lxr/source/external/geant4.3.1/source/g3tog4/src/G4gsrotm.cc

 48         // Construct unit std::vectors 
 49     
 50     G4ThreeVector x(sin(th1r)*cos(phi1r), sin(th1r)*sin(phi1r), cos(th1r)->second;
 51     G4ThreeVector y(sin(th2r)*cos(phi2r), sin(th2r)*sin(phi2r), cos(th2r));
 52     G4ThreeVector z(sin(th3r)*cos(phi3r), sin(th3r)*sin(phi3r), cos(th3r));
 53 
 54         // check for orthonormality and left-handedness
 55 
 56     G4double check = (x.cross(y))*z;
 57     G4double tol = 1.0e-3;
 58         
 59     if (1-abs(check)>tol) {
 60         G4cerr << "Coordinate axes forming rotation matrix "
 61                << irot << " are not orthonormal.(" << 1-abs(check) << ")" 
 62          << G4std::endl;
 63         G4cerr << " thetaX=" << theta1;
 64         G4cerr << " phiX=" << phi1;
 65         G4cerr << " thetaY=" << theta2;
 66         G4cerr << " phiY=" << phi2;
 67         G4cerr << " thetaZ=" << theta3;
 68         G4cerr << " phiZ=" << phi3;
 69         G4cerr << G4std::endl;
 70         G4Exception("G4gsrotm error");
 71     }
 72     else if (1+check<=tol) {
 73         G4cerr << "G4gsrotm warning: coordinate axes forming rotation "
 74                << "matrix " << irot << " are left-handed" << G4std::endl;
 75     }   
 76
 77     G3toG4RotationMatrix* rotp = new G3toG4RotationMatrix;
 78 
 79     rotp->SetRotationMatrixByRow(x, y, z);

  ****************/


  // check for orthonormality and left-handedness
  
  double check = (x.Cross(y)).Dot(z);
  double tol = 1.0e-3;
  ExprEvalInterface & ev = ExprEvalSingleton::instance();
  DDXMLAttribute atts = getAttributeSet();
  
  if (1.0-std::abs(check)>tol) {
    std::cout << "DDLRotationAndReflection Coordinate axes forming rotation matrix "
	      << getDDName(nmspace) 
	      << " are not orthonormal.(tolerance=" << tol 
	      << " check=" << std::abs(check)  << ")" 
	      << std::endl
	      << " thetaX=" << (atts.find("thetaX")->second) 
	      << ' ' << ev.eval(nmspace, atts.find("thetaX")->second)/deg << std::endl
	      << " phiX=" << (atts.find("phiX")->second) 
	      << ' ' << ev.eval(nmspace, atts.find("phiX")->second)/deg << std::endl
	      << " thetaY=" << (atts.find("thetaY")->second) 
	      << ' ' << ev.eval(nmspace, atts.find("thetaY")->second)/deg << std::endl
	      << " phiY=" << (atts.find("phiY")->second)
	      << ' ' << ev.eval(nmspace, atts.find("phiY")->second)/deg << std::endl
	      << " thetaZ=" << (atts.find("thetaZ")->second)
	      << ' ' << ev.eval(nmspace, atts.find("thetaZ")->second)/deg << std::endl
	      << " phiZ=" << (atts.find("phiZ")->second)
	      << ' ' << ev.eval(nmspace, atts.find("phiZ")->second)/deg 
	      << std::endl
	      << "  WAS NOT CREATED!" << std::endl;
    ret = -1;
  }
  else if (1.0+check<=tol) {
    ret = 1;    
  }
  DCOUT_V('P', "DDLRotation::isLeftHanded completed");
  return ret;
}

DD3Vector
DDLRotationAndReflection::makeX(std::string nmspace)
{
  DD3Vector x;
  DDXMLAttribute atts = getAttributeSet();
  if (atts.find("thetaX") != atts.end())
  {
    ExprEvalInterface & ev = ExprEvalSingleton::instance(); 
    double thetaX = ev.eval(nmspace, atts.find("thetaX")->second.c_str());
    double phiX = ev.eval(nmspace, atts.find("phiX")->second.c_str());
    // colx
    x.SetX(sin(thetaX) * cos(phiX));
    x.SetY(sin(thetaX) * sin(phiX));
    x.SetZ(cos(thetaX));
  }
  return x;
}

DD3Vector
DDLRotationAndReflection::makeY(std::string nmspace)
{
  DD3Vector y;
  DDXMLAttribute atts = getAttributeSet();
  if (atts.find("thetaY") != atts.end())
  {
    ExprEvalInterface & ev = ExprEvalSingleton::instance(); 
    double thetaY = ev.eval(nmspace, atts.find("thetaY")->second.c_str());
    double phiY = ev.eval(nmspace, atts.find("phiY")->second.c_str());
      
    // coly
    y.SetX(sin(thetaY) * cos(phiY));
    y.SetY(sin(thetaY) * sin(phiY));
    y.SetZ(cos(thetaY));
  }
  return y;
}

DD3Vector DDLRotationAndReflection::makeZ(std::string nmspace)
{
  DD3Vector z;
  DDXMLAttribute atts = getAttributeSet();
  if (atts.find("thetaZ") != atts.end())
  {
    ExprEvalInterface & ev = ExprEvalSingleton::instance(); 
    double thetaZ = ev.eval(nmspace, atts.find("thetaZ")->second.c_str());
    double phiZ = ev.eval(nmspace, atts.find("phiZ")->second.c_str());
      
    // colz
    z.SetX(sin(thetaZ) * cos(phiZ));
    z.SetY(sin(thetaZ) * sin(phiZ));
    z.SetZ(cos(thetaZ));
  }
  return z;
}

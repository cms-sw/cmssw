#include "DetectorDescription/Parser/src/DDLRotationAndReflection.h"
#include "DetectorDescription/Core/interface/DDRotationMatrix.h"
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDTransform.h"
#include "DetectorDescription/Core/interface/ClhepEvaluator.h"
#include "DetectorDescription/Core/interface/DDUnits.h"
#include "DetectorDescription/Parser/interface/DDLElementRegistry.h"
#include "DetectorDescription/Parser/src/DDXMLElement.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Math/GenVector/Cartesian3D.h"
#include "Math/GenVector/DisplacementVector3D.h"

#include <cmath>
#include <iostream>
#include <map>
#include <utility>

class DDCompactView;

using namespace dd::operators;

DDLRotationAndReflection::DDLRotationAndReflection( DDLElementRegistry* myreg )
  : DDXMLElement( myreg ) 
{}

void
DDLRotationAndReflection::processElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv )
{
  DD3Vector x = makeX(nmspace);
  DD3Vector y = makeY(nmspace);
  DD3Vector z = makeZ(nmspace);

  DDXMLAttribute atts = getAttributeSet();


  if ((name == "Rotation") && isLeftHanded(x, y, z, nmspace) == 0)
  {
    DDRotationMatrix* ddr = new DDRotationMatrix(x, y, z);
    DDRotation ddrot = DDrot(getDDName(nmspace), ddr);
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
    ClhepEvaluator & ev = myRegistry_->evaluator();
    DDRotation ddrot = 
      DDrotReflect(getDDName(nmspace)
		   , ev.eval(nmspace, atts.find("thetaX")->second)
		   , ev.eval(nmspace, atts.find("phiX")->second)
		   , ev.eval(nmspace, atts.find("thetaY")->second)
		   , ev.eval(nmspace, atts.find("phiY")->second)
		   , ev.eval(nmspace, atts.find("thetaZ")->second)
		   , ev.eval(nmspace, atts.find("phiZ")->second));
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
DDLRotationAndReflection::isLeftHanded (const DD3Vector& x, const DD3Vector& y, const DD3Vector& z, const std::string & nmspace)
{
  int ret = 0;

  // check for orthonormality and left-handedness
  
  double check = (x.Cross(y)).Dot(z);
  double tol = 1.0e-3;
  ClhepEvaluator & ev = myRegistry_->evaluator();
  DDXMLAttribute atts = getAttributeSet();
  
  if (1.0-std::abs(check)>tol) {
    std::cout << "DDLRotationAndReflection Coordinate axes forming rotation matrix "
	      << getDDName(nmspace) 
	      << " are not orthonormal.(tolerance=" << tol 
	      << " check=" << std::abs(check)  << ")" 
	      << std::endl
	      << " thetaX=" << (atts.find("thetaX")->second) 
	      << ' ' << CONVERT_TO( ev.eval(nmspace, atts.find("thetaX")->second), deg ) << std::endl
	      << " phiX=" << (atts.find("phiX")->second) 
	      << ' ' << CONVERT_TO( ev.eval(nmspace, atts.find("phiX")->second), deg ) << std::endl
	      << " thetaY=" << (atts.find("thetaY")->second) 
	      << ' ' << CONVERT_TO( ev.eval(nmspace, atts.find("thetaY")->second), deg ) << std::endl
	      << " phiY=" << (atts.find("phiY")->second)
	      << ' ' << CONVERT_TO( ev.eval(nmspace, atts.find("phiY")->second), deg ) << std::endl
	      << " thetaZ=" << (atts.find("thetaZ")->second)
	      << ' ' << CONVERT_TO( ev.eval(nmspace, atts.find("thetaZ")->second), deg ) << std::endl
	      << " phiZ=" << (atts.find("phiZ")->second)
	      << ' ' << CONVERT_TO( ev.eval(nmspace, atts.find("phiZ")->second), deg ) 
	      << std::endl
	      << "  WAS NOT CREATED!" << std::endl;
    ret = -1;
  }
  else if (1.0+check<=tol) {
    ret = 1;    
  }
  return ret;
}

DD3Vector
DDLRotationAndReflection::makeX(const std::string& nmspace)
{
  DD3Vector x;
  DDXMLAttribute atts = getAttributeSet();
  if (atts.find("thetaX") != atts.end())
  {
    ClhepEvaluator & ev = myRegistry_->evaluator(); 
    double thetaX = ev.eval(nmspace, atts.find("thetaX")->second);
    double phiX = ev.eval(nmspace, atts.find("phiX")->second);
    // colx
    x.SetX(sin(thetaX) * cos(phiX));
    x.SetY(sin(thetaX) * sin(phiX));
    x.SetZ(cos(thetaX));
  }
  return x;
}

DD3Vector
DDLRotationAndReflection::makeY(const std::string& nmspace)
{
  DD3Vector y;
  DDXMLAttribute atts = getAttributeSet();
  if (atts.find("thetaY") != atts.end())
  {
    ClhepEvaluator & ev = myRegistry_->evaluator(); 
    double thetaY = ev.eval(nmspace, atts.find("thetaY")->second);
    double phiY = ev.eval(nmspace, atts.find("phiY")->second);
      
    // coly
    y.SetX(sin(thetaY) * cos(phiY));
    y.SetY(sin(thetaY) * sin(phiY));
    y.SetZ(cos(thetaY));
  }
  return y;
}

DD3Vector
DDLRotationAndReflection::makeZ(const std::string& nmspace)
{
  DD3Vector z;
  DDXMLAttribute atts = getAttributeSet();
  if (atts.find("thetaZ") != atts.end())
  {
    ClhepEvaluator & ev = myRegistry_->evaluator(); 
    double thetaZ = ev.eval(nmspace, atts.find("thetaZ")->second);
    double phiZ = ev.eval(nmspace, atts.find("phiZ")->second);
      
    // colz
    z.SetX(sin(thetaZ) * cos(phiZ));
    z.SetY(sin(thetaZ) * sin(phiZ));
    z.SetZ(cos(thetaZ));
  }
  return z;
}

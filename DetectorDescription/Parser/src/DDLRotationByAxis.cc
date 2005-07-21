/***************************************************************************
                          DDLRotationByAxis.cc  -  description
                             -------------------
    begin                : Wed Nov 19, 2003
    email                : case@ucdhep.ucdavis.edu
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *           DDDParser sub-component of DDD                                *
 *                                                                         *
 ***************************************************************************/

namespace std{} using namespace std;

// -------------------------------------------------------------------------
// Includes
// -------------------------------------------------------------------------
#include "DetectorDescription/Parser/interface/DDLRotationByAxis.h"
#include "DetectorDescription/Parser/interface/DDLElementRegistry.h"

// DDCore dependencies
#include "DetectorDescription/Core/interface/DDTransform.h"
#include "DetectorDescription/Base/interface/DDdebug.h"
#include "DetectorDescription/Base/interface/DDException.h"
#include "DetectorDescription/ExprAlgo/interface/ExprEvalSingleton.h"

// CLHEP dependencies
#include "CLHEP/Geometry/Transform3D.h"

#include <string>

//namespace ddl {

// Default constructor
DDLRotationByAxis::DDLRotationByAxis() 
{
}

// Default destructor
DDLRotationByAxis::~DDLRotationByAxis()
{
}

void DDLRotationByAxis::preProcessElement (const std::string& name
					   , const std::string& nmspace)
{
  pNameSpace = nmspace;
  pName = name;
}

void DDLRotationByAxis::processElement (const std::string& name
					, const std::string& nmspace)
{
  DCOUT_V('P', "DDLRotationByAxis::processElement started " << name);

  //  cout << "my parent is : " << parent() << endl;
  DDXMLAttribute atts = getAttributeSet();
  if (parent() != "RotationSequence")
    {
      string axis = atts.find("axis")->second;
      string angle = atts.find("angle")->second;
      
      HepRotation R;
      R = processOne(R, axis, angle);

      DDRotationMatrix* ddr = new DDRotationMatrix(R);
      if (atts.find("name") == atts.end())
	{
	  //how do we make up a ddname! damn_it!
          DDXMLElement * myRealParent = DDLElementRegistry::instance()->getElement(parent());
	  DDName pName = myRealParent->getDDName(nmspace);
	  string tn = pName.name() + string("Rotation");
	  vector<string> names;
	  names.push_back("name");
	  //no need, used already names.push_back("axis");
	  //no need, used already names.push_back("angle");

	  vector<string> values;
	  values.push_back(tn);
	  //no need, used already values.push_back(atts.find("axis")->second);
	  //no need, used already values.push_back(atts.find("angle")->second);
	  clear();
	  loadAttributes(name, names, values, nmspace);
	}
      try {	
	DDRotation rot = DDrot(getDDName(nmspace), ddr);
      } catch (DDException & e) {
	string msg(e.what());
	msg += "\nDDLRotationByAxis failed to create a DDrot.";
	throwError(msg);
      }

      clear();
    }
  else { } //let the parent handle the clearing, etc.

  DCOUT_V('P', "DDLRotationByAxis::processElement completed");
}

HepRotation DDLRotationByAxis::processOne (HepRotation R, string& axis, string& angle)
{
  /** Get the name, axis and angle of the RotationByAxis and do it. 
   */
  
  ExprEvalInterface & ev = ExprEvalSingleton::instance();
  double dAngle = ev.eval(pNameSpace, angle);
  //  HepRotation R;

  if ( axis == "x") {
      R.rotateX(dAngle);      
    }
  else if ( axis == "y" ) 
    {
      R.rotateY(dAngle);      
    }
  else if ( axis =="z" )
      R.rotateZ(dAngle);
  else
    {
      string msg = "\nDDLRotationByAxis invalid axis... you must not have validated XML sources!  Element is ";
      msg += pName;
      throwError(msg);
    }

  return R;
}

//}

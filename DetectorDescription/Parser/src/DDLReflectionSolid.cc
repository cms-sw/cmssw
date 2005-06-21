/***************************************************************************
                          DDLReflectionSolid.cc  -  description
                             -------------------
    begin                : Mon Mar 4, 2002
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
#include "DetectorDescription/DDParser/interface/DDLReflectionSolid.h"
#include "DetectorDescription/DDParser/interface/DDLElementRegistry.h"
#include "DetectorDescription/DDParser/interface/DDXMLElement.h"

// DDCore dependencies
#include "DetectorDescription/DDCore/interface/DDName.h"
#include "DetectorDescription/DDCore/interface/DDSolid.h"
#include "DetectorDescription/DDBase/interface/DDdebug.h"
#include "DetectorDescription/DDBase/interface/DDException.h"

#include "DetectorDescription/DDExprAlgo/interface/ExprEvalSingleton.h"

#include <string>

// Default constructor
DDLReflectionSolid::DDLReflectionSolid()
{
}

// Default desctructor
DDLReflectionSolid::~DDLReflectionSolid()
{
}

// Upon starting a ReflectionSolid element, we need to clear all rSolids.
void DDLReflectionSolid::preProcessElement(const string& name, const string& nmspace)
{
  DDLElementRegistry::getElement("rSolid")->clear();
}

// Upon ending a ReflectionSolid element, call DDCore giving the solid name, and dimensions.
void DDLReflectionSolid::processElement (const string& name, const string& nmspace)
{
  DCOUT_V('P', "DDLReflectionSolid::processElement started");

  // get solid reference:
  DDXMLElement* myrSolid = DDLElementRegistry::getElement("rSolid");

  if (myrSolid->size() != 1)
    {
      cout << "WARNING:  A ReflectionSolid had more than one rSolid.  "
	   << "The first one was used." << endl;
      cout << "The element to look for is " << getDDName(nmspace) << endl;
    }

  try {
    DDSolid solid = DDSolid(myrSolid->getDDName(nmspace));
    DDSolid ddreflsol = DDSolidFactory::reflection(getDDName(nmspace), solid);
  } catch (DDException & e) {
    string msg(e.what());
    msg += "\nDDLReflectionSolid failed call to DDSolidFactory.";
    throwError(msg);
  }

  DDLSolid::setReference(nmspace);

  DCOUT_V('P', "DDLReflectionSolid::processElement completed");
}

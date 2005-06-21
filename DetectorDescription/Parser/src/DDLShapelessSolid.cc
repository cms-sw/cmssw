/***************************************************************************
                          DDLShapelessSolid.cc  -  description
                             -------------------
    begin                : Wed May 15 2002
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
#include "DetectorDescription/DDParser/interface/DDLShapelessSolid.h"
#include "DetectorDescription/DDParser/interface/DDLElementRegistry.h"

// DDCore dependencies
#include "DetectorDescription/DDCore/interface/DDName.h"
#include "DetectorDescription/DDCore/interface/DDSolid.h"
#include "DetectorDescription/DDBase/interface/DDdebug.h"

//#include <strstream>
#include <string>

// Default constructor
DDLShapelessSolid::DDLShapelessSolid()
{
}

// Default desctructor
DDLShapelessSolid::~DDLShapelessSolid()
{
}

void DDLShapelessSolid::preProcessElement(const string& name, const string& nmspace)
{
  DDLElementRegistry::getElement("rSolid")->clear();
}
// Upon ending a ShapelessSolid element, call DDCore giving the box name, and dimensions.
void DDLShapelessSolid::processElement (const string& type, const string& nmspace)
{
  DCOUT_V('P', "DDLShapelessSolid::processElement started");

  try {
    DDSolid dds = DDSolidFactory::shapeless(getDDName(nmspace));
  } catch (DDException & e) {
    string msg(e.what());
    msg += "\nDDLShapelessSolid failed call to DDSolidFactory.";
    throwError(msg);
  }
  DDLSolid::setReference(nmspace);

  DCOUT_V('P', "DDLShapelessSolid::processElement completed");
}

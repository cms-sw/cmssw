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



// -------------------------------------------------------------------------
// Includes
// -------------------------------------------------------------------------
#include "DetectorDescription/Parser/interface/DDLShapelessSolid.h"
#include "DetectorDescription/Parser/interface/DDLElementRegistry.h"

// DDCore dependencies
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Base/interface/DDdebug.h"

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

void DDLShapelessSolid::preProcessElement(const std::string& name, const std::string& nmspace)
{
  DDLElementRegistry::getElement("rSolid")->clear();
}
// Upon ending a ShapelessSolid element, call DDCore giving the box name, and dimensions.
void DDLShapelessSolid::processElement (const std::string& type, const std::string& nmspace)
{
  DCOUT_V('P', "DDLShapelessSolid::processElement started");

  try {
    DDSolid dds = DDSolidFactory::shapeless(getDDName(nmspace));
  } catch (DDException & e) {
    std::string msg(e.what());
    msg += "\nDDLShapelessSolid failed call to DDSolidFactory.";
    throwError(msg);
  }
  DDLSolid::setReference(nmspace);

  DCOUT_V('P', "DDLShapelessSolid::processElement completed");
}

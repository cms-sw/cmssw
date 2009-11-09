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
#include "DDLShapelessSolid.h"
#include "DDLElementRegistry.h"

// DDCore dependencies
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Base/interface/DDdebug.h"

//#include <strstream>

// Default constructor
DDLShapelessSolid::DDLShapelessSolid(  DDLElementRegistry* myreg ) : DDLSolid(myreg)
{
}

// Default desctructor
DDLShapelessSolid::~DDLShapelessSolid()
{
}

void DDLShapelessSolid::preProcessElement(const std::string& name, const std::string& nmspace, DDCompactView& cpv)
{
  myRegistry_->getElement("rSolid")->clear();
}
// Upon ending a ShapelessSolid element, call DDCore giving the box name, and dimensions.
void DDLShapelessSolid::processElement (const std::string& name, const std::string& nmspace, DDCompactView& cpv)
{
  DCOUT_V('P', "DDLShapelessSolid::processElement started");

  DDSolid dds = DDSolidFactory::shapeless(getDDName(nmspace));
    
  DDLSolid::setReference(nmspace, cpv);

  DCOUT_V('P', "DDLShapelessSolid::processElement completed");
}

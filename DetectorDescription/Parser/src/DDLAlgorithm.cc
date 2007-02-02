/***************************************************************************
                          DDLAlgorithm.cc  -  description
                             -------------------
    begin                : Saturday November 29, 2003
    email                : case@ucdhep.ucdavis.edu
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *           DDDParser sub-component of DDD                                *
 *                                                                         *
 ***************************************************************************/



// Parser parts
#include "DetectorDescription/Parser/interface/DDLAlgorithm.h"
#include "DetectorDescription/Parser/interface/DDLVector.h"
#include "DetectorDescription/Parser/interface/DDLMap.h"
#include "DetectorDescription/Parser/interface/DDLElementRegistry.h"
#include "DetectorDescription/Parser/interface/DDXMLElement.h"

// DDCore dependencies
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Base/interface/DDdebug.h"
#include "DetectorDescription/Core/interface/DDNumeric.h"
#include "DetectorDescription/Core/interface/DDString.h"
#include "DetectorDescription/Core/interface/DDVector.h"
#include "DetectorDescription/Core/interface/DDMap.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithm.h"
#include "DetectorDescription/Base/interface/DDException.h"

#include "DetectorDescription/Algorithm/interface/DDAlgorithmHandler.h"

// CLHEP dependencies
#include "CLHEP/Units/SystemOfUnits.h"
#include "DetectorDescription/ExprAlgo/interface/ExprEvalSingleton.h"

#include <string>
#include <sstream>

DDLAlgorithm::DDLAlgorithm()
{
}

DDLAlgorithm::~DDLAlgorithm()
{
}

void DDLAlgorithm::preProcessElement (const std::string& name, const std::string& nmspace)
{
  DDLElementRegistry::getElement("Vector")->clear();
}


void DDLAlgorithm::processElement (const std::string& name, const std::string& nmspace)
{
  DCOUT_V('P',"DDLAlgorithm::processElement started");

  DDXMLElement* myNumeric        = DDLElementRegistry::getElement("Numeric");
  DDXMLElement* myString         = DDLElementRegistry::getElement("String");
  DDXMLElement* myVector         = DDLElementRegistry::getElement("Vector");
  DDXMLElement* myMap            = DDLElementRegistry::getElement("Map");
  DDXMLElement* myrParent        = DDLElementRegistry::getElement("rParent");

  DDName algoName(getDDName(nmspace));  
  DDLogicalPart lp(DDName(myrParent->getDDName(nmspace)));
  DDXMLAttribute atts;

  // handle all Numeric elements in the Algorithm.
  DDNumericArguments nArgs;
  size_t i = 0;
  for (; i < myNumeric->size(); ++i)
    {
      atts = myNumeric->getAttributeSet(i);
      nArgs[atts.find("name")->second] = ExprEvalSingleton::instance().eval(nmspace, atts.find("value")->second);
    }

  DDStringArguments sArgs;
  for (i = 0; i < myString->size(); ++i)
    {
      atts = myString->getAttributeSet(i);
      sArgs[atts.find("name")->second] = atts.find("value")->second;
    }

  DDAlgorithmHandler handler;
  atts = getAttributeSet();
  try {
    DDLVector* tv= dynamic_cast<DDLVector*> (myVector);
    DDLMap* tm= dynamic_cast<DDLMap*> (myMap);
    handler.initialize( algoName, lp, nArgs, tv->getMapOfVectors(), tm->getMapOfMaps(), sArgs, tv->getMapOfStrVectors() );
    handler.execute();
  }
  catch (const DDException & e) {
    std::string msg = e.what();
    msg += "\nDDLParser (DDLAlgorithm) caught DDException.";
    msg += "\nReference: <" + name + " name=\"" + (atts.find("name")->second);
    msg += "\">";
    msg += " in namespace " + nmspace + "\n";
    throwError(msg);  
  }
  catch ( ... ) {
    std::string msg("DDLParser (DDLAlgorithm) failed with unknown error.");
    msg+= "Reference: <" + name + " name=\"" + atts.find("name")->second;
    msg+= "\" nEntries=\"" + atts.find("nEntries")->second + "\">";
    msg+= " in namespace " + nmspace + "\n";
    throwError(msg);
  }

  myString->clear();
  myNumeric->clear();
  myVector->clear();
  myMap->clear();
  myrParent->clear();
  clear();

  DCOUT_V('P',"DDLAlgorithm::processElement(...)");
}


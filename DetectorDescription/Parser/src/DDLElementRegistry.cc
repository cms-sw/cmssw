/***************************************************************************
                          DDLElementRegistry.cc  -  description
                             -------------------
    begin                : Wed Oct 24 2001
    email                : case@ucdhep.ucdavis.edu
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *           DDDParser sub-component of DDD                                *
 *  Nov 25, 2003 : note that comments on DDLRotation are for future        *
 *                 changes which break backward compatibility.             *
 *                                                                         *
 ***************************************************************************/



// -------------------------------------------------------------------------
// Includes
// -------------------------------------------------------------------------
// DDL parts
#include "DDLAlgorithm.h"
#include "DDLAlgoPosPart.h"
#include "DDLBooleanSolid.h"
#include "DDLBox.h"
#include "DDLCompositeMaterial.h"
#include "DDLCone.h"
#include "DDLDivision.h"
#include "DDLElementRegistry.h"
#include "DDLElementaryMaterial.h"
#include "DDLLogicalPart.h"
#include "DDLMap.h"
#include "DDLNumeric.h"
#include "DDLPolyGenerator.h"
#include "DDLPosPart.h"
#include "DDLPseudoTrap.h"
#include "DDLReflectionSolid.h"
#include "DDLRotationByAxis.h"
#include "DDLRotationAndReflection.h"
#include "DDLRotationSequence.h"
#include "DDLShapelessSolid.h" 
#include "DDLSpecPar.h"
#include "DDLSphere.h"
#include "DDLString.h"
#include "DDLTorus.h"
#include "DDLTrapezoid.h"
#include "DDLTubs.h"
#include "DDLVector.h"

// DDCore dependencies
#include "DetectorDescription/Base/interface/DDdebug.h"

#include <iostream>

// -------------------------------------------------------------------------
// Static member initialization
// -------------------------------------------------------------------------
//DDLElementRegistry* DDLElementRegistry::instance_ = 0;

// Note that an XML element can not be all spaces or all asterisks, etc. :-) 
// so we are safe to use this.
//std::string DDLElementRegistry::defaultElement_ = "*****";

// -------------------------------------------------------------------------
// Constructor/Destructor
// -------------------------------------------------------------------------

DDLElementRegistry::DDLElementRegistry()
{ }

DDLElementRegistry::~DDLElementRegistry() { 

}

// -------------------------------------------------------------------------
// Implementation
// -------------------------------------------------------------------------

// This initializes and acts as the singleton instance of DDLElementRegistry.
DDLElementRegistry* DDLElementRegistry::instance()
{
  static DDLElementRegistry reg;
  static bool isInit=false;
  if (!isInit) {
    isInit=true;
    instance()->registerElement("***", new DDXMLElement);
  }
  return &reg;
  /*  
      if (instance_ == 0)
      {
      instance_ = new DDLElementRegistry;
      
      DDXMLElement* defaultElement = new DDXMLElement; 
      instance()->registerElement(defaultElement_, defaultElement); 
      }
      return instance_;
  */  
}

DDXMLElement* DDLElementRegistry::getElement(const std::string& name)
{
  DCOUT_V('P',"DDLElementRegistry::getElementRegistry(" << name << ")"); 

  DDXMLElement* myret = instance()->DDXMLElementRegistry::getElement(name);

  if (myret == NULL)
    {

      // Make the Solid handlers and register them.
      if (name == "Box")
        {
          myret = new DDLBox;
        }
      else if (name == "Cone")
        {
          myret =  new DDLCone;
        }
      else if (name == "Polyhedra" || name == "Polycone")
        {
          myret = new DDLPolyGenerator;
        }
      else if (name == "Trapezoid" || name == "Trd1")
        {
          myret = new DDLTrapezoid;
        }
      else if (name == "PseudoTrap")
        {
      	  myret = new DDLPseudoTrap;
        }
      else if (name == "Tubs" || name == "Tube" || name == "TruncTubs")
        {
      	  myret = new DDLTubs;
        }
      else if (name == "Torus")
        {
      	  myret = new DDLTorus;
        }
      else if (name == "ReflectionSolid")
        {
      	  myret = new DDLReflectionSolid;
        }
      else if (name == "UnionSolid" || name == "SubtractionSolid"
      	       || name == "IntersectionSolid")
        {
      	  myret = new DDLBooleanSolid;
        }
      else if (name == "ShapelessSolid")
        {
      	  myret = new DDLShapelessSolid;
        }

      //  LogicalParts, Positioners, Materials, Rotations, Reflections
      //  and Specific (Specified?) Parameters
      else if (name == "PosPart")
        {
      	  myret = new DDLPosPart;
        }
      else if (name == "AlgoPosPart")
        {
      	  myret = new DDLAlgoPosPart;
        }
      else if (name == "CompositeMaterial")
        {
      	  myret = new DDLCompositeMaterial;
        }
      else if (name == "ElementaryMaterial")
        {
      	  myret = new DDLElementaryMaterial;
        }
      else if (name == "LogicalPart")
        {
      	  myret = new DDLLogicalPart;
        }
      else if (name == "ReflectionRotation" || name == "Rotation" )
        {
      	  myret = new DDLRotationAndReflection;
        }
      else if (name == "SpecPar")
        {
      	  myret = new DDLSpecPar;
        }
      else if (name == "Sphere")
	{
	  myret = new DDLSphere;
	}
      else if (name == "RotationSequence")
	{
	  myret = new DDLRotationSequence;
	}
      else if (name == "RotationByAxis")
	{
	  myret = new DDLRotationByAxis;
	}
      // Special, need them around.
      else if (name == "SpecParSection") {
	myret = new DDXMLElement(true);
      }
      else if (name == "Vector") {
        myret = new DDLVector;
      }
      else if (name == "Map") {
        myret = new DDLMap;
      }
      else if (name == "String") {
        myret = new DDLString;
      }
      else if (name == "Numeric") {
        myret = new DDLNumeric;
      }
      else if (name == "Algorithm") {
        myret = new DDLAlgorithm;
      }
      else if (name == "Division") {
	myret = new DDLDivision;
      }

      // Supporting Cast of elements.
      //  All elements which simply accumulate attributes which are then used
      //  by one of the above elements.
      else if (name == "MaterialFraction" || name == "ParE" || name == "ParS"
	       || name == "RZPoint" || name == "PartSelector"
	       || name == "Parameter" || name == "ZSection"
	       || name == "Translation" 
	       || name == "rSolid" || name == "rMaterial" 
	       || name == "rParent" || name == "rChild"
	       || name == "rRotation" || name == "rReflectionRotation"
	       || name == "DDDefinition" )
        {
          myret = new DDXMLElement;
        }


      //  IF it is a new element return a default XMLElement which processes nothing.
      //  Since there are elements in the XML which require no processing, they
      //  can all use the same DDXMLElement which defaults to anything.  A validated
      //  XML document (i.e. validated with XML Schema) will work properly.
      //  As of 8/16/2002:  Elements like LogicalPartSection and any other *Section
      //  XML elements of the DDLSchema are taken care of by this.
      else
	{
    	  myret = instance()->DDXMLElementRegistry::getElement("***");
	  DCOUT_V('P',  "WARNING:  The default (DDLElementRegistry)  was used for "
		  << name << " since there was no specific handler." << std::endl);
    	}
      
      // Actually register the thing
      instance()->registerElement(name, myret);
    }

  return myret;
}

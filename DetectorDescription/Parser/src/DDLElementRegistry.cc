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

#include "DetectorDescription/Parser/src/DDLAlgorithm.h"
#include "DetectorDescription/Parser/src/DDLBooleanSolid.h"
#include "DetectorDescription/Parser/src/DDLBox.h"
#include "DetectorDescription/Parser/src/DDLCompositeMaterial.h"
#include "DetectorDescription/Parser/src/DDLCone.h"
#include "DetectorDescription/Parser/src/DDLDivision.h"
#include "DetectorDescription/Parser/src/DDLElementaryMaterial.h"
#include "DetectorDescription/Parser/src/DDLEllipticalTube.h"
#include "DetectorDescription/Parser/src/DDLEllipsoid.h"
#include "DetectorDescription/Parser/src/DDLLogicalPart.h"
#include "DetectorDescription/Parser/src/DDLMap.h"
#include "DetectorDescription/Parser/src/DDLNumeric.h"
#include "DetectorDescription/Parser/src/DDLOrb.h"
#include "DetectorDescription/Parser/src/DDLParallelepiped.h"
#include "DetectorDescription/Parser/src/DDLPolyGenerator.h"
#include "DetectorDescription/Parser/src/DDLPosPart.h"
#include "DetectorDescription/Parser/src/DDLPseudoTrap.h"
#include "DetectorDescription/Parser/src/DDLReflectionSolid.h"
#include "DetectorDescription/Parser/src/DDLRotationByAxis.h"
#include "DetectorDescription/Parser/src/DDLRotationAndReflection.h"
#include "DetectorDescription/Parser/src/DDLRotationSequence.h"
#include "DetectorDescription/Parser/src/DDLShapelessSolid.h" 
#include "DetectorDescription/Parser/src/DDLSpecPar.h"
#include "DetectorDescription/Parser/src/DDLSphere.h"
#include "DetectorDescription/Parser/src/DDLString.h"
#include "DetectorDescription/Parser/src/DDLTorus.h"
#include "DetectorDescription/Parser/src/DDLTrapezoid.h"
#include "DetectorDescription/Parser/src/DDLTubs.h"
#include "DetectorDescription/Parser/src/DDLVector.h"

#include "DetectorDescription/Base/interface/DDdebug.h"
#include "DetectorDescription/Parser/interface/DDLElementRegistry.h"

#include <iostream>

DDLElementRegistry::DDLElementRegistry( void )
{}

DDLElementRegistry::~DDLElementRegistry( void ) 
{
  // Complicated cleanup.  I keep track of DDXMLElements that have
  // already been deleted using this vector.  Then delete them one-by-one.
  std::vector<DDXMLElement*> toDelete;
  for( RegistryMap::const_iterator it = registry_.begin(), end = registry_.end(); it != end; ++it )
  {
    std::vector<DDXMLElement*>::const_iterator deleteIt = std::find( toDelete.begin(), toDelete.end(), it->second );
    if( deleteIt == toDelete.end())
    {
      toDelete.push_back( it->second );
      delete it->second;
    }
  }
}

// -------------------------------------------------------------------------
// Implementation
// -------------------------------------------------------------------------
DDXMLElement*
DDLElementRegistry::getElement( const std::string& name )
{
  DCOUT_V('P',"myRegistry_->getElementRegistry(" << name << ")"); 

  //  DDXMLElement* myret = instance()->DDXMLElementRegistry::getElement(name);
  RegistryMap::iterator it = registry_.find(name);
  //   std::cout << "it found name? "<< name << " " ;
  //   if (it != registry_.end() ) std::cout << "yes"; else std::cout << "no";
  //   std::cout << std::endl;
  //   std::cout << "there are " << registry_.size() << " elements-types so far." << std::endl; 
  DDXMLElement* myret = NULL;
  if( it != registry_.end())
  {
    myret = it->second;
  } else {
    //    std::cout << " making first and only " << name << std::endl;
    // Make the Solid handlers and register them.
    if (name == "Box")
    {
      myret = new DDLBox(this);
    }
    else if (name == "Cone")
    {
      myret =  new DDLCone(this);
    }
    else if (name == "Polyhedra" || name == "Polycone")
    {
      myret = new DDLPolyGenerator(this);
    }
    else if (name == "Trapezoid" || name == "Trd1")
    {
      myret = new DDLTrapezoid(this);
    }
    else if (name == "PseudoTrap")
    {
      myret = new DDLPseudoTrap(this);
    }
    else if (name == "Tubs" || name == "Tube" || name == "TruncTubs")
    {
      myret = new DDLTubs(this);
    }
    else if (name == "Torus")
    {
      myret = new DDLTorus(this);
    }
    else if (name == "ReflectionSolid")
    {
      myret = new DDLReflectionSolid(this);
    }
    else if (name == "UnionSolid" || name == "SubtractionSolid"
	     || name == "IntersectionSolid")
    {
      myret = new DDLBooleanSolid(this);
    }
    else if (name == "ShapelessSolid")
    {
      myret = new DDLShapelessSolid(this);
    }
    else if (name == "Sphere")
    {
      myret = new DDLSphere(this);
    }
    else if (name == "Orb")
    {
      myret = new DDLOrb(this);
    }
    else if (name == "EllipticalTube")
    {
      myret = new DDLEllipticalTube(this);
    }
    else if (name == "Ellipsoid")
    {
      myret = new DDLEllipsoid(this);
    }
    else if (name == "Sphere")
    {
      myret = new DDLParallelepiped(this);
    }

    //  LogicalParts, Positioners, Materials, Rotations, Reflections
    //  and Specific (Specified?) Parameters
    else if (name == "PosPart")
    {
      myret = new DDLPosPart(this);
    }
    else if (name == "CompositeMaterial")
    {
      myret = new DDLCompositeMaterial(this);
    }
    else if (name == "ElementaryMaterial")
    {
      myret = new DDLElementaryMaterial(this);
    }
    else if (name == "LogicalPart")
    {
      myret = new DDLLogicalPart(this);
    }
    else if (name == "ReflectionRotation" || name == "Rotation" )
    {
      myret = new DDLRotationAndReflection(this);
    }
    else if (name == "SpecPar")
    {
      myret = new DDLSpecPar(this);
    }
    else if (name == "RotationSequence")
    {
      myret = new DDLRotationSequence(this);
    }
    else if (name == "RotationByAxis")
    {
      myret = new DDLRotationByAxis(this);
    }
    // Special, need them around.
    else if (name == "SpecParSection") {
      myret = new DDXMLElement(this, true);
    }
    else if (name == "Vector") {
      myret = new DDLVector(this);
    }
    else if (name == "Map") {
      myret = new DDLMap(this);
    }
    else if (name == "String") {
      myret = new DDLString(this);
    }
    else if (name == "Numeric") {
      myret = new DDLNumeric(this);
    }
    else if (name == "Algorithm") {
      myret = new DDLAlgorithm(this);
    }
    else if (name == "Division") {
      myret = new DDLDivision(this);
    }

    // Supporting Cast of elements.
    //  All elements which simply accumulate attributes which are then used
    //  by one of the above elements.
    else if (name == "MaterialFraction"
	     || name == "RZPoint" || name == "PartSelector"
	     || name == "Parameter" || name == "ZSection"
	     || name == "Translation" 
	     || name == "rSolid" || name == "rMaterial" 
	     || name == "rParent" || name == "rChild"
	     || name == "rRotation" || name == "rReflectionRotation"
	     || name == "DDDefinition" )
    {
      myret = new DDXMLElement(this);
    }

    //  IF it is a new element return a default XMLElement which processes nothing.
    //  Since there are elements in the XML which require no processing, they
    //  can all use the same DDXMLElement which defaults to anything.  A validated
    //  XML document (i.e. validated with XML Schema) will work properly.
    //  As of 8/16/2002:  Elements like LogicalPartSection and any other *Section
    //  XML elements of the DDLSchema are taken care of by this.
    else
    {
      //	myret = instance()->DDXMLElementRegistry::getElement("***");
	
      myret = new DDXMLElement(this);
      //	std::cout << "about to register a " << "***" << std::endl;
      // 	registry_["***"] = myret;
      // 	DCOUT_V('P',  "WARNING:  The default (DDLElementRegistry)  was used for "
      // 		<< name << " since there was no specific handler." << std::endl);
      // 	return myret;
    }
    
    // Actually register the thing
    //   instance()->registerElement(name, myret);
    //      std::cout << "about to register a " << name << std::endl;
    registry_[name] = myret;
  }
  return myret;
}

const std::string&
DDLElementRegistry::getElementName( DDXMLElement* theElement ) const
{
  for( RegistryMap::const_iterator it = registry_.begin(), end = registry_.end(); it != end; ++it )
    if( it->second == theElement )
      return it->first;
  return registry_.find( "***" )->first;
}

template class DDI::Singleton<DDLElementRegistry>;


/***************************************************************************
                          DDLRotationSequence.cc  -  description
                             -------------------
    begin                : Friday November 14, 2003
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
#include "DDLRotationSequence.h"
#include "DDLElementRegistry.h"

// DDCore dependencies
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Base/interface/DDdebug.h"
#include "DetectorDescription/Base/interface/DDException.h"
#include "DetectorDescription/Base/interface/DDRotationMatrix.h"
#include "DetectorDescription/Core/interface/DDTransform.h"

// CLHEP dependencies
//#include "CLHEP/Geometry/Transform3D.h"

#include <string>

// Default constructor
DDLRotationSequence::DDLRotationSequence() 
{
}

// Default destructor
DDLRotationSequence::~DDLRotationSequence()
{
}

void DDLRotationSequence::preProcessElement (const std::string& name, const std::string& nmspace)
{
  DDLElementRegistry::getElement("RotationByAxis")->clear();
}


void DDLRotationSequence::processElement (const std::string& name, const std::string& nmspace)
{

  DCOUT_V('P', "DDLRotationSequence::processElement started " << name);

  /** Get the name, axis and angle of each Rotate child and make this the rotation. 
   */

  DDLRotationByAxis* myRotations = 
    dynamic_cast <DDLRotationByAxis * > (DDLElementRegistry::getElement("RotationByAxis"));
  DDXMLAttribute atts;

  DDRotationMatrix R;
  for (size_t i = 0; i < myRotations->size(); ++i)
    {
      atts = myRotations->getAttributeSet(i);
      R = myRotations->processOne(R, atts.find("axis")->second, atts.find("angle")->second);
    }
    
  DDRotationMatrix* ddr = new DDRotationMatrix(R);
  DDRotation rot = DDrot(getDDName(nmspace), ddr);

  myRotations->clear();
  clear();

  DCOUT_V('P', "DDLRotationSequence::processElement completed");
}

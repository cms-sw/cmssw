#include "DetectorDescription/Parser/src/DDLRotationSequence.h"
#include "DetectorDescription/Core/interface/DDRotationMatrix.h"
#include "DetectorDescription/Core/interface/DDTransform.h"
#include "DetectorDescription/Parser/interface/DDLElementRegistry.h"
#include "DetectorDescription/Parser/src/DDLRotationByAxis.h"
#include "DetectorDescription/Parser/src/DDXMLElement.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <cstddef>
#include <map>
#include <utility>

class DDCompactView;

DDLRotationSequence::DDLRotationSequence( DDLElementRegistry* myreg )
  : DDLRotationByAxis( myreg ) 
{}

void
DDLRotationSequence::preProcessElement( const std::string& name, const std::string& nmspace,
					DDCompactView& cpv )
{
  myRegistry_->getElement("RotationByAxis")->clear();
}

void
DDLRotationSequence::processElement( const std::string& name, const std::string& nmspace,
				     DDCompactView& cpv )
{
  /** Get the name, axis and angle of each Rotate child and make this the rotation. 
   */

  std::shared_ptr<DDLRotationByAxis> myRotations =
    std::static_pointer_cast<DDLRotationByAxis>(myRegistry_->getElement("RotationByAxis"));
  DDXMLAttribute atts;

  DDRotationMatrix R;
  for (size_t i = 0; i < myRotations->size(); ++i)
  {
    atts = myRotations->getAttributeSet(i);
    R = myRotations->processOne(R, atts.find("axis")->second, atts.find("angle")->second);
  }
  
  DDRotation rot = DDrot(getDDName(nmspace), std::make_unique<DDRotationMatrix>( R ));

  myRotations->clear();
  clear();
}

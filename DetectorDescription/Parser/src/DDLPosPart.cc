#include "DetectorDescription/Parser/src/DDLPosPart.h"
#include "DetectorDescription/Core/interface/DDRotationMatrix.h"
#include "DetectorDescription/Core/interface/DDTranslation.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDTransform.h"
#include "DetectorDescription/Core/interface/ClhepEvaluator.h"
#include "DetectorDescription/Parser/interface/DDLElementRegistry.h"
#include "DetectorDescription/Parser/src/DDXMLElement.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <map>
#include <utility>

DDLPosPart::DDLPosPart( DDLElementRegistry* myreg )
  : DDXMLElement( myreg )
{}

// Upon encountering a PosPart, store the label, simple.
// Just in case some left-over Rotation has not been cleared, make sure
// that it is cleared.  
// I commented out the others because the last element
// that made use of them should have cleared them.
void
DDLPosPart::preProcessElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv )
{
  // Clear out child elements.
  myRegistry_->getElement("Rotation")->clear();
  myRegistry_->getElement("ReflectionRotation")->clear();
}

// Upon encountering the end tag of the PosPart we should have in the meantime
// hit two rLogicalPart calls and one of Rotation or rRotation and a Translation.
// So, retrieve them and make the call to DDCore.
void
DDLPosPart::processElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv )
{
  // get all internal elements.
  auto myParent     = myRegistry_->getElement("rParent");
  auto myChild      = myRegistry_->getElement("rChild");
  auto myTranslation= myRegistry_->getElement("Translation");
  auto myDDLRotation= myRegistry_->getElement("Rotation");
  auto myrRotation  = myRegistry_->getElement("rRotation");
  auto myDDLRefl    = myRegistry_->getElement("ReflectionRotation");
  auto myrRefl      = myRegistry_->getElement("rReflectionRotation");
  // FIXME!!! add in the new RotationByAxis element...

  // At this time, PosPart is becoming the most complex of the elements.
  // For simply reflections/rotations we have 4 possible internal "components"
  // to the PosPart.  We take them in the following order of priority
  //     rRotation, Rotation, rReflectionRotation, ReflectionRotation.
  //
  // The idea in the following if-else-if is that no matter
  // what was used inside the PosPart element, the order in which we
  // will look for and use an internal element is:
  // rRotation, Rotation, ReflectionRotation, rReflectionRotation.
  // If it falls through here, a default call will result in a nameless 
  // "identity" rotation being passed to DDCore.
  DDName rotn;
  if (myrRotation->size() > 0){
    rotn = myrRotation->getDDName(nmspace);
  }
  else if (myDDLRotation->size() > 0) {
    // The assumption here is that the Rotation element created 
    // a DDRotation already, and so we can use this as an rRotation
    // just provide DDCore with the name of the one just added... 
    // How to handle name conflicts? OVERWRITTEN by DDCore for now.
    rotn = myDDLRotation->getDDName(nmspace);
  }
  else if (myDDLRefl->size() > 0) {
    // The assumption is that a ReflectionRotation has been created and therefore 
    // we can refer to it as the rotation associated with this PosPart.
    // we can further assume that the namespace is the same as this PosPart.
    rotn = myDDLRefl->getDDName(nmspace);
  }
  else if (myrRefl->size() > 0) {
    rotn = myrRefl->getDDName(nmspace);
  }

  ClhepEvaluator & ev = myRegistry_->evaluator();

  double x = 0.0, y = 0.0, z = 0.0;
  if (myTranslation->size() > 0)
  {
    const DDXMLAttribute & atts = myTranslation->getAttributeSet();
    x = ev.eval(nmspace, atts.find("x")->second);
    y = ev.eval(nmspace, atts.find("y")->second);
    z = ev.eval(nmspace, atts.find("z")->second);
  }

  std::unique_ptr<DDRotation> myDDRotation;
  // if rotation is named ...
  if ( !rotn.name().empty() && !rotn.ns().empty() ) {
    myDDRotation = std::make_unique<DDRotation>(rotn);
  } else { 
    // rotn is not assigned a name anywhere therefore the DDPos assumes the identity matrix.
    myDDRotation = std::make_unique<DDRotation>(DDName(std::string("identity"),std::string("generatedForDDD")));
    // if the identity is not yet defined, then...
    if ( !myDDRotation->isValid() ) {
      myDDRotation = DDrotPtr( DDName( std::string( "identity" ), std::string( "generatedForDDD" )), std::make_unique<DDRotationMatrix>());
    }
  }


  DDTranslation myDDTranslation(x, y, z);

  const DDXMLAttribute & atts = getAttributeSet();
  std::string copyno = "";
  if (atts.find("copyNumber") != atts.end())
    copyno = atts.find("copyNumber")->second;
    
  cpv.position(DDLogicalPart(myChild->getDDName(nmspace))
	       , DDLogicalPart(myParent->getDDName(nmspace))
	       , copyno
	       , myDDTranslation
	       , *myDDRotation);
  
  // clear all "children" and attributes
  myParent->clear();
  myChild->clear();
  myTranslation->clear();
  myDDLRotation->clear();
  myrRotation->clear();
  myDDLRefl->clear();
  myrRefl->clear();

  // after a pos part is done, we know we can clear it.
  clear();
}

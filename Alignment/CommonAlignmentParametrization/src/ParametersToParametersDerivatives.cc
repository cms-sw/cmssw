/** \file ParametersToParametersDerivatives.cc
 *
 *  $Date: 2007/03/12 21:28:48 $
 *  $Revision: 1.5 $
 */

#include "Alignment/CommonAlignmentParametrization/interface/ParametersToParametersDerivatives.h"

#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignmentParameters.h"
#include "Alignment/CommonAlignmentParametrization/interface/FrameToFrameDerivative.h"
#include "Alignment/CommonAlignmentParametrization/interface/AlignmentParametersFactory.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

// already in header:
//#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"


//__________________________________________________________________________________________________
ParametersToParametersDerivatives
::ParametersToParametersDerivatives(const Alignable &component, const Alignable &mother)
  : isOK_(component.alignmentParameters() && mother.alignmentParameters())
{
  if (isOK_) {
    isOK_ = this->init(component, component.alignmentParameters()->type(),
		       mother,    mother   .alignmentParameters()->type());
  }
}

//__________________________________________________________________________________________________
bool ParametersToParametersDerivatives::init(const Alignable &component, int typeComponent,
					     const Alignable &mother, int typeMother)
{
  using namespace AlignmentParametersFactory; // for kRigidBody etc.
  if ((typeMother    == kRigidBody || typeMother    == kRigidBody4D) &&
      (typeComponent == kRigidBody || typeComponent == kRigidBody4D)) {
    return this->initRigidRigid(component, mother);
  } else {
    edm::LogError("Alignment") << "@SUB=ParametersToParametersDerivatives::init"
			       << "Mother " << parametersTypeName(parametersType(typeMother))
			       << ", component " << parametersTypeName(parametersType(typeComponent))
			       << ": not supported.";
    return false;
  }

}

//__________________________________________________________________________________________________
bool ParametersToParametersDerivatives::initRigidRigid(const Alignable &component,
						       const Alignable &mother)
{
  // simply frame to frame!
  FrameToFrameDerivative f2fDerivMaker;
  derivatives_ = f2fDerivMaker.frameToFrameDerivative(&component, &mother);

  return true;
}


//__________________________________________________________________________________________________
double ParametersToParametersDerivatives::operator() (unsigned int indParMother,
						      unsigned int indParComp) const
{
  // Do range checks?
  return derivatives_[indParMother][indParComp];
} 

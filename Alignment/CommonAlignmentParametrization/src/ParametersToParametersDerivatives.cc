/** \file ParametersToParametersDerivatives.cc
 *
 *  $Date: 2010/12/14 01:08:26 $
 *  $Revision: 1.2 $
 */

#include "Alignment/CommonAlignmentParametrization/interface/ParametersToParametersDerivatives.h"

#include "CondFormats/Alignment/interface/Definitions.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignmentParameters.h"
#include "Alignment/CommonAlignmentParametrization/interface/FrameToFrameDerivative.h"
#include "Alignment/CommonAlignmentParametrization/interface/AlignmentParametersFactory.h"
#include "Alignment/CommonAlignmentParametrization/interface/BowedSurfaceAlignmentDerivatives.h"
#include "Alignment/CommonAlignmentParametrization/interface/TwoBowedSurfacesAlignmentParameters.h"

#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
// already in header:
// #include "DataFormats/Math/interface/AlgebraicROOTObjects.h"

//_________________________________________________________________________________________________
ParametersToParametersDerivatives
::ParametersToParametersDerivatives(const Alignable &component, const Alignable &mother)
  : isOK_(component.alignmentParameters() && mother.alignmentParameters())
{
  if (isOK_) {
    isOK_ = this->init(component, component.alignmentParameters()->type(),
		       mother,    mother   .alignmentParameters()->type());
  }
}

//_________________________________________________________________________________________________
bool ParametersToParametersDerivatives::init(const Alignable &component, int typeComponent,
					     const Alignable &mother, int typeMother)
{
  using namespace AlignmentParametersFactory; // for kRigidBody etc.
  if ((typeMother    == kRigidBody || typeMother    == kRigidBody4D) &&
      (typeComponent == kRigidBody || typeComponent == kRigidBody4D)) {
    return this->initRigidRigid(component, mother);
  } else if ((typeMother == kRigidBody || typeMother == kRigidBody4D) &&
	     typeComponent == kBowedSurface) {
    return this->initBowedRigid(component, mother);
  } else if ((typeMother == kRigidBody || typeMother == kRigidBody4D) &&
	     typeComponent == kTwoBowedSurfaces) {
    return this->init2BowedRigid(component, mother);
  } else {
    // missing: mother with bows and component without, i.e. having 'common' bow parameters
    edm::LogError("Alignment") << "@SUB=ParametersToParametersDerivatives::init"
			       << "Mother " << parametersTypeName(parametersType(typeMother))
			       << ", component " << parametersTypeName(parametersType(typeComponent))
			       << ": not supported.";
    return false;
  }

}

//_________________________________________________________________________________________________
bool ParametersToParametersDerivatives::initRigidRigid(const Alignable &component,
						       const Alignable &mother)
{
  // See G. Flucke's presentation from  20 Feb 2007
  // https://indico.cern.ch/contributionDisplay.py?contribId=15&sessionId=1&confId=10930
  // and C. Kleinwort's one from 14 Feb 2013
  // https://indico.cern.ch/contributionDisplay.py?contribId=14&sessionId=1&confId=224472

  FrameToFrameDerivative f2f;
  // frame2frame returns dcomponent/dmother for both being rigid body, so we have to invert
  AlgebraicMatrix66 m(asSMatrix<6,6>(f2f.frameToFrameDerivative(&component, &mother)));
  
  if (m.Invert()) { // now matrix is d(rigid_mother)/d(rigid_component)
    // copy to TMatrix
    derivatives_.ResizeTo(6,6);
    derivatives_.SetMatrixArray(m.begin());
    return true;
  } else {
    return false;
  }
}

//_________________________________________________________________________________________________
bool ParametersToParametersDerivatives::initBowedRigid(const Alignable &component,
						       const Alignable &mother)
{
  // component is bowed surface, mother rigid body
  FrameToFrameDerivative f2f;
  // frame2frame returns dcomponent/dmother for both being rigid body, so we have to invert
  AlgebraicMatrix66 rigM2rigC(asSMatrix<6,6>(f2f.frameToFrameDerivative(&component,&mother)));
  if (rigM2rigC.Invert()) { // now matrix is d(rigid_mother)/d(rigid_component)
    const double halfWidth  = 0.5 * component.surface().width();
    const double halfLength = 0.5 * component.surface().length();
    const AlgebraicMatrix69 m(this->dRigid_dBowed(rigM2rigC, halfWidth, halfLength));

    // copy to TMatrix
    derivatives_.ResizeTo(6,9);
    derivatives_.SetMatrixArray(m.begin());
    
    return true;
  } else {
    return false;
  }
}

//_________________________________________________________________________________________________
bool ParametersToParametersDerivatives::init2BowedRigid(const Alignable &component,
							const Alignable &mother)
{
  // component is two bowed surfaces, mother rigid body
  const TwoBowedSurfacesAlignmentParameters *aliPar = 
    dynamic_cast<TwoBowedSurfacesAlignmentParameters*>(component.alignmentParameters());

  if (!aliPar) {
    edm::LogError("Alignment") << "@SUB=ParametersToParametersDerivatives::init2BowedRigid"
			       << "dynamic_cast to TwoBowedSurfacesAlignmentParameters failed.";
    return false;
  }

  // We treat the two surfaces as independent objects, i.e.
  // 1) get the global position of each surface, depending on the ySplit value,
  const double ySplit = aliPar->ySplit();
  const double halfWidth   = 0.5 * component.surface().width();
  const double halfLength  = 0.5 * component.surface().length();
  const double halfLength1 = 0.5 * (halfLength + ySplit);
  const double halfLength2 = 0.5 * (halfLength - ySplit);
  const double yM1 = 0.5 * (ySplit - halfLength); // y_mean of surface 1
  const double yM2 = yM1 + halfLength;            // y_mean of surface 2
  // The sensor positions and orientations could be adjusted using
  // TwoBowedSurfacesDeformation attached to the component,
  // but that should be 2nd order effect.
  const align::GlobalPoint posSurf1(component.surface().toGlobal(align::LocalPoint(0.,yM1,0.)));
  const align::GlobalPoint posSurf2(component.surface().toGlobal(align::LocalPoint(0.,yM2,0.)));

  // 2) get derivatives for both,
  // getDerivative(..) returns dcomponent/dmother for both being rigid body
  FrameToFrameDerivative f2fMaker;
  AlgebraicMatrix66 f2fSurf1(f2fMaker.getDerivative(component.globalRotation(),
                                                    mother.globalRotation(),
                                                    posSurf1, mother.globalPosition()));
  AlgebraicMatrix66 f2fSurf2(f2fMaker.getDerivative(component.globalRotation(),
                                                    mother.globalRotation(),
                                                    posSurf2, mother.globalPosition()));
  // We have to invert matrices to get d(rigid_mother)/d(rigid_component):
  if (!f2fSurf1.Invert() || !f2fSurf2.Invert()) return false; // bail out if bad inversion
  // Now get d(rigid_mother)/d(bowed_component):
  const AlgebraicMatrix69 derivs1(this->dRigid_dBowed(f2fSurf1, halfWidth, halfLength1));
  const AlgebraicMatrix69 derivs2(this->dRigid_dBowed(f2fSurf2, halfWidth, halfLength2));

  // 3) fill the common matrix by merging the two.
  typedef ROOT::Math::SMatrix<double,6,18,ROOT::Math::MatRepStd<double,6,18> > AlgebraicMatrix6_18;
  AlgebraicMatrix6_18 derivs;
  derivs.Place_at(derivs1, 0, 0); // left half
  derivs.Place_at(derivs2, 0, 9); // right half

  // copy to TMatrix
  derivatives_.ResizeTo(6, 18);
  derivatives_.SetMatrixArray(derivs.begin());

  return true;
}

//_________________________________________________________________________________________________
ParametersToParametersDerivatives::AlgebraicMatrix69
ParametersToParametersDerivatives::dRigid_dBowed(const AlgebraicMatrix66 &dRigidM2dRigidC,
						 double halfWidth, double halfLength)
{
  typedef BowedSurfaceAlignmentDerivatives BowedDerivs;
  const double gammaScale = BowedDerivs::gammaScale(2.*halfWidth, 2.*halfLength);

  // 'dRigidM2dRigidC' is dmother/dcomponent for both being rigid body
  // final matrix will be dmother/dcomponent for mother as rigid body, component with bows
  // 1st index (row) is parameter of the mother (0..5),
  // 2nd index (column) that of component (0..8):
  AlgebraicMatrix69 derivs;
  if (0. == gammaScale || 0. == halfWidth || 0. == halfLength) {
    isOK_ = false; // bad input - we would have to devide by that in the following!
    edm::LogError("Alignment") << "@SUB=ParametersToParametersDerivatives::dRigid_dBowed"
			       << "Some zero length as input.";
    return derivs;
  }

  for (unsigned int iRow = 0; iRow < AlgebraicMatrix69::kRows; ++iRow) {
    // loop on 6 rigid body parameters of mother
    // First copy the common rigid body part, e.g.:
    // (0,0): du_moth/du_comp, (0,1): dv_moth/du_comp, (1,2): dw_moth/dv_comp
    for (unsigned int iCol = 0; iCol < 3; ++iCol) { // 3 movements of component
      derivs(iRow, iCol) = dRigidM2dRigidC(iRow, iCol);  
    }

    // Now we have to take care of order, signs and scales for rotation-like parameters,
    // see CMS AN-2011/531:
    // slopeX = w10 = -halfWidth * beta
    // => dpar_m/dslopeX_comp = dpar_m/d(-hw * beta_comp) = -(dpar_m/dbeta_comp)/hw
    derivs(iRow, 3) = -dRigidM2dRigidC(iRow, 4)/halfWidth; 
    // slopeY = w10 = +halfLength * alpha
    // => dpar_m/dslopeY_comp = dpar_m/d(+hl * alpha_comp) = (dpar_m/dalpha_comp)/hl
    derivs(iRow, 4) = dRigidM2dRigidC(iRow, 3)/halfLength;
    // rotZ = gammaScale * gamma
    // => dpar_m/drotZ_comp = dpar_m/d(gamma_comp * gscale) = (dpar_m/dgamma)/gscale
    derivs(iRow, 5) = dRigidM2dRigidC(iRow, 5)/gammaScale;

    // Finally, sensor internals like their curvatures have no influence on mother:
    for (unsigned int iCol = 6; iCol < AlgebraicMatrix69::kCols; ++iCol) {
      derivs(iRow, iCol) = 0.; // 3 sagittae of component
    }
  }

  return derivs;
}

//_________________________________________________________________________________________________
double ParametersToParametersDerivatives::operator() (unsigned int indParMother,
						      unsigned int indParComp) const
{
  // Do range checks?
  return derivatives_(indParMother, indParComp);
} 

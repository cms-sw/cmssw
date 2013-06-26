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
  // simply frame to frame!
  FrameToFrameDerivative f2fDerivMaker;
  AlgebraicMatrix66 m(asSMatrix<6,6>(f2fDerivMaker.frameToFrameDerivative(&component, &mother)));

  // copy to TMatrix
  derivatives_.ResizeTo(6,6);
  derivatives_.SetMatrixArray(m.begin());

  return true;
}

//_________________________________________________________________________________________________
bool ParametersToParametersDerivatives::initBowedRigid(const Alignable &component,
						       const Alignable &mother)
{
  // component is bowed surface, mother rigid body
  FrameToFrameDerivative f2fMaker;
  const AlgebraicMatrix66 f2f(asSMatrix<6,6>(f2fMaker.frameToFrameDerivative(&component,&mother)));
  const double halfWidth  = 0.5 * component.surface().width();
  const double halfLength = 0.5 * component.surface().length();
  const AlgebraicMatrix69 m(this->dBowed_dRigid(f2f, halfWidth, halfLength));

  // copy to TMatrix
  derivatives_.ResizeTo(6,9);
  derivatives_.SetMatrixArray(m.begin());

  return true;
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
  FrameToFrameDerivative f2fMaker;
  const AlgebraicMatrix66 f2fSurf1(f2fMaker.getDerivative(component.globalRotation(),
                                                          mother.globalRotation(),
                                                          posSurf1, mother.globalPosition()));
  const AlgebraicMatrix66 f2fSurf2(f2fMaker.getDerivative(component.globalRotation(),
                                                          mother.globalRotation(),
                                                          posSurf2, mother.globalPosition()));
  const AlgebraicMatrix69 derivs1(this->dBowed_dRigid(f2fSurf1, halfWidth, halfLength1));
  const AlgebraicMatrix69 derivs2(this->dBowed_dRigid(f2fSurf2, halfWidth, halfLength2));

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
ParametersToParametersDerivatives::dBowed_dRigid(const AlgebraicMatrix66 &f2f,
						 double halfWidth, double halfLength) const
{
  typedef BowedSurfaceAlignmentDerivatives BowedDerivs;
  const double gammaScale = BowedDerivs::gammaScale(2.*halfWidth, 2.*halfLength);

  // 1st index (column) is parameter of the mother (<6),
  // 2nd index (row) that of component (<9):
  AlgebraicMatrix69 derivs;

  for (unsigned int iRow = 0; iRow < 6; ++iRow) { // 6 rigid body parameters of mother
    // First copy the common rigid body part, e.g.:
    // - (0,0): du_comp/du_moth
    // - (0,1): dv_comp/du_moth
    // - (1,2): dw_comp/dv_moth
    for (unsigned int iCol = 0; iCol < 3; ++iCol) { // 3 movements of component
      derivs(iRow, iCol) = f2f(iRow, iCol);  
    }

    // Now we have to take care of order and scales for rotation-like parameters:
    // slopeX -> halfWidth * beta
    derivs(iRow, 3) = halfWidth  * f2f(iRow, 4); // = dslopeX_c/dpar_m = hw * db_c/dpar_m
    // slopeY -> halfLength * alpha
    derivs(iRow, 4) = halfLength * f2f(iRow, 3); // = dslopeY_c/dpar_m = hl * da_c/dpar_m
    // rotZ -> gammaScale * gamma
    derivs(iRow, 5) = gammaScale * f2f(iRow, 5); // = drotZ_c/dpar_m = gscale * dg_c/dpar_m

    // Finally, movements and rotations have no influence on surface internals:
    for (unsigned int iCol = 6; iCol < 9; ++iCol) { // 3 sagittae of component
      derivs(iRow, iCol) = 0.;  
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

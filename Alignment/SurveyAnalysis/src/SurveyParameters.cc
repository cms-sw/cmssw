#include "Alignment/SurveyAnalysis/interface/SurveyParameters.h"
#include "Alignment/CommonAlignmentParametrization/interface/AlignmentParametersFactory.h"
#include "FWCore/Utilities/interface/Exception.h"

SurveyParameters::SurveyParameters(Alignable* object, const AlgebraicVector& par, const AlgebraicSymMatrix& cov)
    : AlignmentParameters(object, par, cov) {}

void SurveyParameters::apply() {
  throw cms::Exception("BadInheritance") << "SurveyParameters::apply(): Not implemented.";
}

int SurveyParameters::type() const { return AlignmentParametersFactory::kSurvey; }

AlignmentParameters* SurveyParameters::clone(const AlgebraicVector&, const AlgebraicSymMatrix&) const {
  throw cms::Exception("BadInheritance") << "SurveyParameters::clone(): Not implemented.";
  return nullptr;
}

AlignmentParameters* SurveyParameters::cloneFromSelected(const AlgebraicVector&, const AlgebraicSymMatrix&) const {
  throw cms::Exception("BadInheritance") << "SurveyParameters::cloneFromSelected(): Not implemented.";

  return nullptr;
}

AlgebraicMatrix SurveyParameters::derivatives(const TrajectoryStateOnSurface&, const AlignableDetOrUnitPtr&) const {
  throw cms::Exception("BadInheritance") << "SurveyParameters::derivatives(): Not implemented.";

  return AlgebraicMatrix();
}

AlgebraicMatrix SurveyParameters::selectedDerivatives(const TrajectoryStateOnSurface&,
                                                      const AlignableDetOrUnitPtr&) const {
  throw cms::Exception("BadInheritance") << "SurveyParameters::selectedDerivatives(): Not implemented.";

  return AlgebraicMatrix();
}

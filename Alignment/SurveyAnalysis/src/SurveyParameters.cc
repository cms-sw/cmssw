#include "Alignment/SurveyAnalysis/interface/SurveyParameters.h"

SurveyParameters::SurveyParameters(Alignable* object,
				   const AlgebraicVector& par,
				   const AlgebraicSymMatrix& cov):
  AlignmentParameters(object, par, cov)
{
}

AlignmentParameters* SurveyParameters::clone(const AlgebraicVector&,
					     const AlgebraicSymMatrix&) const
{
  return 0;
}

AlignmentParameters* SurveyParameters::cloneFromSelected(const AlgebraicVector&,
							 const AlgebraicSymMatrix&) const
{
  return 0;
}

AlgebraicMatrix SurveyParameters::derivatives(const TrajectoryStateOnSurface&,
					      const AlignableDetOrUnitPtr& ) const
{
  return AlgebraicMatrix();
}

AlgebraicMatrix SurveyParameters::selectedDerivatives(const TrajectoryStateOnSurface&,
						      const AlignableDetOrUnitPtr& ) const
{
  return AlgebraicMatrix();
}

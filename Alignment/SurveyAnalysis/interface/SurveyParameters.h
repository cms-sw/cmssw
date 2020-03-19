#ifndef Alignment_SurveyAnalysis_SurveyParameters_h
#define Alignment_SurveyAnalysis_SurveyParameters_h

/** \class SurveyParameters
 *
 *  Alignment parameters for survey.
 *  Inheriting from AlignmentParameters is fake, just to attach it to an Alignable,
 *  re-using the AlignmentParameters data member. (Should look for another solution...)
 *
 *  $Date: 2007/05/09 12:42:03 $
 *  $Revision: 1.2 $
 *  \author Chung Khim Lae
 */

#include "Alignment/CommonAlignment/interface/AlignmentParameters.h"

class SurveyParameters : public AlignmentParameters {
public:
  /// Set the alignable, parameters, covariance in base class.
  /// No user variables, default is all parameters are selected and valid.
  SurveyParameters(Alignable*, const AlgebraicVector& par, const AlgebraicSymMatrix& cov);

  /// apply not implemented
  void apply() override;
  int type() const override;

  /// Cloning not implemented.
  AlignmentParameters* clone(const AlgebraicVector&, const AlgebraicSymMatrix&) const override;

  /// Cloning not implemented.
  AlignmentParameters* cloneFromSelected(const AlgebraicVector&, const AlgebraicSymMatrix&) const override;

  /// Derivatives not implemented.
  AlgebraicMatrix derivatives(const TrajectoryStateOnSurface&, const AlignableDetOrUnitPtr&) const override;

  /// Derivatives not implemented.
  AlgebraicMatrix selectedDerivatives(const TrajectoryStateOnSurface&, const AlignableDetOrUnitPtr&) const override;
};

#endif

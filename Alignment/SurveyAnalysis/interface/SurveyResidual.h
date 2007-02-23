#ifndef Alignment_SurveyAnalysis_SurveyResidual_h
#define Alignment_SurveyAnalysis_SurveyResidual_h

/** \class SurveyResidual
 *
 *  Class to find the residuals for survey constraint alignment.
 *
 *  For more info, please refer to
 *    http://www.pha.jhu.edu/~ntran/cms/fpix_survey.pdf
 *
 *  $Date: 2007/01/25 $
 *  $Revision: 1 $
 *  \author Chung Khim Lae
 */

#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"
#include "Alignment/SurveyAnalysis/interface/SurveyCalculation.h"

class Alignable;
class AlignableSurface;

class SurveyResidual
{
  typedef AlignableObjectId::AlignableObjectIdType AlignableType;

  public:

  /// Find matrix needed to rotate alignable's mother from its nominal
  /// surface to its current surface.
  survey::RotMatrix diffMomRot() const;

  /// Constructor from an alignable whose residuals are to be found.
  /// The type of residuals (panel, disc etc.) is given by AlignableType.
  /// Set bias to true for biased residuals.
  /// Default is to find unbiased residuals.
  SurveyResidual(
		 const Alignable&,
		 AlignableType,    // level at which residuals are found 
		 bool bias = false // true for biased residuals
		 );

  /// Find residual for the alignable in local frame.
  /// Returns a vector of 6 numbers: first 3 are linear, last 3 are angular.
  AlgebraicVector sensorResidual() const;

  /// Find residuals in local frame for points on the alignable
  /// (current - nominal vectors).
  std::vector<LocalVector> pointsResidual() const;

  private:

  /// Find the terminal sisters of an alignable.
  /// bias = true to include itself in the list.
  void findSisters(
		   const Alignable*,
		   bool bias
		   );

  /// Find the nominal and current vectors.
  void calculate(
		 const Alignable&
		 );
		 
  // Cache some values for calculation

  const AlignableSurface& theSurface; // current surface

  std::vector<const Alignable*> theSisters; // list of final daughters for
                                            // finding mother's position

  GlobalPoint theNominalMomPos; // mother's pos from survey
  GlobalPoint theCurrentMomPos; // current mother's pos

  survey::Vectors theNominalVs; // nominal points from mother's pos
  survey::Vectors theCurrentVs; // current points rotated to nominal surface
};

#endif

#ifndef Alignment_SurveyAnalysis_SurveyResidual_h
#define Alignment_SurveyAnalysis_SurveyResidual_h

/** \class SurveyResidual
 *
 *  Class to find the residuals for survey constraint alignment.
 *
 *  For more info, please refer to
 *    http://www.pha.jhu.edu/~gritsan/cms/cms-note-survey.pdf
 *
 *  $Date: 2007/06/24 01:08:20 $
 *  $Revision: 1.3 $
 *  \author Chung Khim Lae
 */

#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"
#include "Alignment/CommonAlignment/interface/Utilities.h"

class Alignable;
class AlignableSurface;

class SurveyResidual
{
  typedef AlignableObjectId::AlignableObjectIdType StructureType;

  public:

  /// Constructor from an alignable whose residuals are to be found.
  /// The type of residuals (panel, disc etc.) is given by AlignableType.
  /// Set bias to true for biased residuals.
  /// Default is to find unbiased residuals.
  SurveyResidual(
		 const Alignable&,
		 StructureType,    // level at which residuals are found 
		 bool bias = false // true for biased residuals
		 );

  /// Find residual for the alignable in local frame.
  /// Returns a vector of 6 numbers: first 3 are linear, last 3 are angular.
  AlgebraicVector sensorResidual() const;

  /// Find residuals in local frame for points on the alignable
  /// (current - nominal vectors).
  align::LocalVectors pointsResidual() const;

  /// Get inverse of survey covariance wrt given structure type in constructor.
  AlgebraicSymMatrix inverseCovariance() const;

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

  const Alignable* theMother; // mother that matches the structure type
                              // given in constructor

  std::vector<const Alignable*> theSisters; // list of final daughters for
                                            // finding mother's position

  align::GlobalVectors theNominalVs; // nominal points from mother's pos
  align::GlobalVectors theCurrentVs; // current points rotated to nominal surf

  align::ErrorMatrix theInverseCovariance;
};

#endif

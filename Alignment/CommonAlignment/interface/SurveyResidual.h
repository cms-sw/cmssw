#ifndef Alignment_CommonAlignment_SurveyResidual_h
#define Alignment_CommonAlignment_SurveyResidual_h

/** \class SurveyResidual
 *
 *  Class to find the residuals for survey constraint alignment.
 *
 *  For more info, please refer to
 *    http://www.pha.jhu.edu/~gritsan/cms/cms-note-survey.pdf
 *
 *  $Date: 2008/11/26 10:21:09 $
 *  $Revision: 1.7 $
 *  \author Chung Khim Lae
 */

#include "Alignment/CommonAlignment/interface/StructureType.h"
#include "Alignment/CommonAlignment/interface/Utilities.h"

class Alignable;
class AlignableSurface;

class SurveyResidual
{
  public:

  /// Constructor from an alignable whose residuals are to be found.
  /// The type of residuals (panel, disc etc.) is given by StructureType.
  /// Set bias to true for biased residuals.
  /// Default is to find unbiased residuals.
  SurveyResidual(
		 const Alignable&,
		 align::StructureType, // level at which residuals are found 
		 bool bias = false     // true for biased residuals
		 );

  /// Check if survey residual is valid (theMother != 0).
  /// This check must be done before calling the other methods so that
  /// calculations can be performed correctly.
  inline bool valid() const;

  /// Find residual for the alignable in local frame.
  /// Returns a vector based on the alignable's dof.
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

  const Alignable* theMother; // mother that matches the structure type
                              // given in constructor

  const AlignableSurface& theSurface; // current surface

  const std::vector<bool>& theSelector; // flags for selected parameters

  std::vector<const Alignable*> theSisters; // list of final daughters for
                                            // finding mother's position

  align::GlobalVectors theNominalVs; // nominal points from mother's pos
  align::GlobalVectors theCurrentVs; // current points rotated to nominal surf

  align::ErrorMatrix theCovariance;
};

bool SurveyResidual::valid() const
{
  return theMother != 0;
}

#endif

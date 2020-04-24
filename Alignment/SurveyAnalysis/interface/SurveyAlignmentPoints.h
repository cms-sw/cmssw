#ifndef Alignment_SurveyAnalysis_SurveyAlignmentPoints_h
#define Alignment_SurveyAnalysis_SurveyAlignmentPoints_h

/** \class SurveyAlignmentPoints
 *
 *  Survey alignment using point residuals.
 *
 *  The local residuals for survey points are found for each sensor.
 *  The alignment parameters are found using the HIP algorithm.
 *
 *  $Date: 2007/05/03 20:58:58 $
 *  $Revision: 1.3 $
 *  \author Chung Khim Lae
 */

#include "Alignment/SurveyAnalysis/interface/SurveyAlignment.h"

class SurveyAlignmentPoints:
  public SurveyAlignment
{
  public:

  /// Constructor to set the sensors and residual levels in base class.
  SurveyAlignmentPoints(
			const align::Alignables& sensors,
			const std::vector<align::StructureType>& levels
			);

  protected:

  /// Find the alignment parameters for all sensors.
  void findAlignPars(
			     bool bias = false // true for biased residuals
			     ) override;
};

#endif

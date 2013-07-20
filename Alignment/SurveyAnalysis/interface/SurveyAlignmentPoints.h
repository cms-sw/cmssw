#ifndef Alignment_SurveyAnalysis_SurveyAlignmentPoints_h
#define Alignment_SurveyAnalysis_SurveyAlignmentPoints_h

/** \class SurveyAlignmentPoints
 *
 *  Survey alignment using point residuals.
 *
 *  The local residuals for survey points are found for each sensor.
 *  The alignment parameters are found using the HIP algorithm.
 *
 *  $Date: 2007/10/08 16:38:03 $
 *  $Revision: 1.4 $
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
  virtual void findAlignPars(
			     bool bias = false // true for biased residuals
			     );
};

#endif

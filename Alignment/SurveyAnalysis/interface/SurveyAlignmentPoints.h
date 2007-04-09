#ifndef Alignment_SurveyAnalysis_SurveyAlignmentPoints_h
#define Alignment_SurveyAnalysis_SurveyAlignmentPoints_h

/** \class SurveyAlignmentPoints
 *
 *  Survey alignment using point residuals.
 *
 *  The local residuals for survey points are found for each sensor.
 *  The alignment parameters are found using the HIP algorithm.
 *
 *  $Date: 2007/03/14 18:05:35 $
 *  $Revision: 1.1 $
 *  \author Chung Khim Lae
 */

#include "Alignment/SurveyAnalysis/interface/SurveyAlignment.h"

class SurveyAlignmentPoints:
  public SurveyAlignment
{
  public:

  /// Constructor to set the sensors in base class.
  SurveyAlignmentPoints(
			const std::vector<Alignable*>& sensors
			);

  protected:

  /// Find the alignment parameters for all sensors.
  virtual void findAlignPars(
			     bool bias = false // true for biased residuals
			     );
};

#endif

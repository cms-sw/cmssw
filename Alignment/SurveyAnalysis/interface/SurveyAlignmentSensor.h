#ifndef Alignment_SurveyAnalysis_SurveyAlignmentSensor_h
#define Alignment_SurveyAnalysis_SurveyAlignmentSensor_h

/** \class SurveyAlignmentSensor
 *
 *  Survey alignment using sensor residual.
 *
 *  The residual (dRx, dRy, dRz, dWx, dWy, dWz) is found for each sensor.
 *  The sensor shifted by this amount during each iteration.
 *
 *  $Date: 2007/10/08 16:38:03 $
 *  $Revision: 1.4 $
 *  \author Chung Khim Lae
 */

#include "Alignment/SurveyAnalysis/interface/SurveyAlignment.h"

class SurveyAlignmentSensor:
  public SurveyAlignment
{
  public:

  /// Constructor to set the sensors and residual levels in base class.
  SurveyAlignmentSensor(
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

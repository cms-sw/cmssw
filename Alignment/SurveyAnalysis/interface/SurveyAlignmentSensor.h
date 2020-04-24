#ifndef Alignment_SurveyAnalysis_SurveyAlignmentSensor_h
#define Alignment_SurveyAnalysis_SurveyAlignmentSensor_h

/** \class SurveyAlignmentSensor
 *
 *  Survey alignment using sensor residual.
 *
 *  The residual (dRx, dRy, dRz, dWx, dWy, dWz) is found for each sensor.
 *  The sensor shifted by this amount during each iteration.
 *
 *  $Date: 2007/05/03 20:58:58 $
 *  $Revision: 1.3 $
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
  void findAlignPars(
			     bool bias = false // true for biased residuals
			     ) override;
};

#endif

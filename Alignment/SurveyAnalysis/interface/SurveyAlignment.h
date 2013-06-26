#ifndef Alignment_SurveyAnalysis_SurveyAlignment_h
#define Alignment_SurveyAnalysis_SurveyAlignment_h

/** \class SurveyAlignment
 *
 *  Alignment using only survey info (no tracks) as a proof of principle.
 *
 *  $Date: 2007/10/08 16:38:03 $
 *  $Revision: 1.5 $
 *  \author Chung Khim Lae
 */

#include "Alignment/CommonAlignment/interface/StructureType.h"
#include "Alignment/CommonAlignment/interface/Utilities.h"

class SurveyAlignment
{
  protected:

  public:

  /// Constructor to set the sensors and residual levels.
  SurveyAlignment(
		  const align::Alignables& sensors,
		  const std::vector<align::StructureType>& levels
		  );

  virtual ~SurveyAlignment() {}

  /// Run the iteration: find residuals, write to output, shift sensors.
  void iterate(
	       unsigned int nIteration,     // number of iterations
	       const std::string& fileName, // name of output file
	       bool bias = false            // true for biased residuals
	       );

  protected:

  /// Find the alignment parameters for all sensors.
  virtual void findAlignPars(
			     bool bias = false // true for biased residuals
			     ) = 0;

  /// Apply the alignment parameters to all sensors.
  virtual void shiftSensors();

  const align::Alignables& theSensors;
  const std::vector<align::StructureType>& theLevels;
};

#endif

#ifndef Alignment_SurveyAnalysis_SurveyAlignment_h
#define Alignment_SurveyAnalysis_SurveyAlignment_h

/** \class SurveyAlignment
 *
 *  Alignment using only survey info (no tracks) as a proof of principle.
 *
 *  $Date: 2007/04/09 03:55:28 $
 *  $Revision: 1.3 $
 *  \author Chung Khim Lae
 */

#include <string>
#include <vector>

#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"

class Alignable;

class SurveyAlignment
{
  protected:

  typedef AlignableObjectId::AlignableObjectIdType StructureType;

  public:

  /// Constructor to set the sensors and residual levels.
  SurveyAlignment(
		  const std::vector<Alignable*>& sensors,
		  const std::vector<StructureType>& levels
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

  const std::vector<Alignable*>& theSensors;
  const std::vector<StructureType>& theLevels;
};

#endif

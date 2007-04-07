#ifndef Alignment_SurveyAnalysis_SurveyAlignment_h
#define Alignment_SurveyAnalysis_SurveyAlignment_h

/** \class SurveyAlignment
 *
 *  Alignment using only survey info (no tracks) as a proof of principle.
 *
 *  $Date: 2007/03/14 18:05:35 $
 *  $Revision: 1.1 $
 *  \author Chung Khim Lae
 */

#include <string>
#include <vector>

class Alignable;

class SurveyAlignment
{
  public:

  /// Constructor to set the sensors.
  SurveyAlignment(
		  const std::vector<Alignable*>& sensors
		  );

  virtual ~SurveyAlignment() {}

  /// Run the iteration: find residuals, write to output, shift sensors.
  void iterate(
	       unsigned int nIteration,    // number of iterations
	       const std::string& fileName // name of output file
	       );

  protected:

  /// Find the alignment parameters for all sensors.
  virtual void findAlignPars() = 0;

  /// Apply the alignment parameters to all sensors.
  virtual void shiftSensors();

  const std::vector<Alignable*>& theSensors;
};

#endif

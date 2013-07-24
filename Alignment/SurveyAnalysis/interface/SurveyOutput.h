#ifndef Alignment_SurveyAnalysis_SurveyOutput_h
#define Alignment_SurveyAnalysis_SurveyOutput_h

/** \class SurveyOutput
 *
 *  Write variables to ntuple for survey analysis.
 *
 *  $Date: 2007/03/14 18:05:35 $
 *  $Revision: 1.1 $
 *  \author Chung Khim Lae
 */

#include <vector>

#include "TFile.h"

class Alignable;

class SurveyOutput
{
  public:

  SurveyOutput(
	       const std::vector<Alignable*>&,
	       const std::string& fileName
	       );

  /// write out variables
  void write(
	     unsigned int iter // iteration number
	     );

  private:

  const std::vector<Alignable*>& theAlignables;

  TFile theFile;
};

#endif

#ifndef Alignment_SurveyAnalysis_SurveyOutput_h
#define Alignment_SurveyAnalysis_SurveyOutput_h

/** \class SurveyOutput
 *
 *  Write variables to ntuple for survey analysis.
 *
 *  $Date: 2007/01/09 $
 *  $Revision: 1 $
 *  \author Chung Khim Lae
 */

#include <vector>

#include "TFile.h"

#include "Alignment/CommonAlignment/interface/Utilities.h"

class Alignable;

class SurveyOutput {
public:
  SurveyOutput(const align::Alignables&, const std::string& fileName);

  /// write out variables
  void write(unsigned int iter  // iteration number
  );

private:
  const align::Alignables& theAlignables;

  TFile theFile;
};

#endif

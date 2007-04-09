#ifndef Alignment_SurveyAnalysis_SurveyTest_h
#define Alignment_SurveyAnalysis_SurveyTest_h

/** \class SurveyTest
 *
 *  Analyser module for testing.
 *
 *  $Date: 2007/04/07 01:58:49 $
 *  $Revision: 1.2 $
 *  \author Chung Khim Lae
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"

class SurveyTest:
  public edm::EDAnalyzer
{
  public:

  SurveyTest(const edm::ParameterSet&);

  virtual void beginJob(const edm::EventSetup&);

  virtual void analyze(const edm::Event&, const edm::EventSetup&) {}

  private:

  void getTerminals(std::vector<Alignable*>& terminals, Alignable* ali);

  bool theBiasFlag; // true for unbiased residuals

  unsigned int theIterations; // number of iterations

  std::string theAlgorithm;  // points or sensor residual
  std::string theOutputFile; // name of output file
};

#endif

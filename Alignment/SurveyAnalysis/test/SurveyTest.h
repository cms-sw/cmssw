#ifndef Alignment_SurveyAnalysis_SurveyTest_h
#define Alignment_SurveyAnalysis_SurveyTest_h

/** \class SurveyTest
 *
 *  Analyser module for testing.
 *
 *  $Date: 2007/03/14 18:05:36 $
 *  $Revision: 1.1 $
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

  unsigned int theIterations; // number of iterations

  std::string theAlgorithm;  // points or sensor residual
  std::string theOutputFile; // name of output file
};

#endif

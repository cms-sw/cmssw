#ifndef Alignment_SurveyAnalysis_SurveyTest_h
#define Alignment_SurveyAnalysis_SurveyTest_h

/** \class SurveyTest
 *
 *  Analyser module for testing.
 *
 *  $Date: 2007/04/09 04:11:13 $
 *  $Revision: 1.4 $
 *  \author Chung Khim Lae
 */

#include <vector>

#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

class Alignable;

class SurveyTest:
  public edm::EDAnalyzer
{
  public:

  SurveyTest(const edm::ParameterSet&);

  virtual void beginJob(const edm::EventSetup&);

  virtual void analyze(const edm::Event&, const edm::EventSetup&) {}

  private:

  void getTerminals(std::vector<Alignable*>& terminals, Alignable* ali);

  bool theBiasFlag; // true for biased residuals

  unsigned int theIterations; // number of iterations

  std::string theAlgorithm;  // points or sensor residual
  std::string theOutputFile; // name of output file

  std::vector<AlignableObjectId::AlignableObjectIdType> theHierarchy;
};

#endif

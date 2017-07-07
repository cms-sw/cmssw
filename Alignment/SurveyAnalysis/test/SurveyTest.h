#ifndef Alignment_SurveyAnalysis_SurveyTest_h
#define Alignment_SurveyAnalysis_SurveyTest_h

/** \class SurveyTest
 *
 *  Analyser module for testing.
 *
 *  $Date: 2007/10/10 20:54:07 $
 *  $Revision: 1.6 $
 *  \author Chung Khim Lae
 */

#include "Alignment/CommonAlignment/interface/StructureType.h"
#include "Alignment/CommonAlignment/interface/Utilities.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

class SurveyTest:
  public edm::EDAnalyzer
{
  public:

  SurveyTest(const edm::ParameterSet&);

  void beginJob() override;

  void analyze(const edm::Event&, const edm::EventSetup&) override {}

  private:

  void getTerminals(align::Alignables& terminals, Alignable* ali);

  bool theBiasFlag; // true for biased residuals

  unsigned int theIterations; // number of iterations

  std::string theAlgorithm;  // points or sensor residual
  std::string theOutputFile; // name of output file

  std::vector<align::StructureType> theHierarchy;
};

#endif

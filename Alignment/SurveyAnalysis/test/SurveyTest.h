#ifndef Alignment_SurveyAnalysis_SurveyTest_h
#define Alignment_SurveyAnalysis_SurveyTest_h

/** \class SurveyTest
 *
 *  Analyser module for testing.
 *
 *  $Date: 2007/01/29 $
 *  $Revision: 1 $
 *  \author Chung Khim Lae
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"

namespace edm { class Event; class EventSetup; class ParameterSet; }

class SurveyTest:
  public edm::EDAnalyzer
{
  public:

  SurveyTest(const edm::ParameterSet&);

  Alignable* create(const std::string&);

  virtual void beginJob(const edm::EventSetup&);

  virtual void analyze(const edm::Event&, const edm::EventSetup&);

  virtual void endJob();

  private:

  edm::ParameterSet theConfig;

  std::vector<Alignable*> theSensors;
  std::vector<SurveyDet*> theSurveys;
};

#endif

#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/SurveyAnalysis/interface/SurveyAlignmentPoints.h"
#include "Alignment/SurveyAnalysis/interface/SurveyAlignmentSensor.h"
#include "Alignment/SurveyAnalysis/interface/SurveyInputBase.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SurveyTest.h"

SurveyTest::SurveyTest(const edm::ParameterSet& cfg):
  theIterations( cfg.getParameter<unsigned int>("iterator") ),
  theAlgorithm ( cfg.getParameter<std::string>("algorith") ),
  theOutputFile( cfg.getParameter<std::string>("fileName") )
{
}

void SurveyTest::beginJob(const edm::EventSetup&)
{
  Alignable* det = SurveyInputBase::detector();

  std::vector<Alignable*> sensors;

  getTerminals(sensors, det);

  std::map<std::string, SurveyAlignment*> algos;

  algos["points"] = new SurveyAlignmentPoints(sensors);
  algos["sensor"] = new SurveyAlignmentSensor(sensors);

  algos[theAlgorithm]->iterate(theIterations, theOutputFile);

  for (std::map<std::string, SurveyAlignment*>::iterator i = algos.begin();
       i != algos.end(); ++i) delete i->second;
}

void SurveyTest::getTerminals(std::vector<Alignable*>& terminals,
			      Alignable* ali)
{
  const std::vector<Alignable*>& comp = ali->components();

  unsigned int nComp = comp.size();

  if (nComp > 0)
    for (unsigned int i = 0; i < nComp; ++i)
    {
      getTerminals(terminals, comp[i]);
    }
  else
    terminals.push_back(ali);
}

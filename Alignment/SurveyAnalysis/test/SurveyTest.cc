#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"
#include "Alignment/SurveyAnalysis/interface/SurveyAlignmentPoints.h"
#include "Alignment/SurveyAnalysis/interface/SurveyAlignmentSensor.h"
#include "Alignment/SurveyAnalysis/interface/SurveyInputBase.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Alignment/SurveyAnalysis/test/SurveyTest.h"

SurveyTest::SurveyTest(const edm::ParameterSet& cfg):
  theBiasFlag( cfg.getUntrackedParameter<bool>("bias", false) ),
  theIterations( cfg.getParameter<unsigned int>("iterator") ),
  theAlgorithm ( cfg.getParameter<std::string>("algorith") ),
  theOutputFile( cfg.getParameter<std::string>("fileName") )
{
  typedef std::vector<std::string> Strings;

  const Strings& hierarchy = cfg.getParameter<Strings>("hierarch");

  for (unsigned int l = 0; l < hierarchy.size(); ++l)
  {
    theHierarchy.push_back(AlignableObjectId::stringToId(hierarchy[l]) );
  }
}

void SurveyTest::beginJob()
{
  Alignable* det = SurveyInputBase::detector();

  std::vector<Alignable*> sensors;

  getTerminals(sensors, det);

  std::map<std::string, SurveyAlignment*> algos;

  algos["points"] = new SurveyAlignmentPoints(sensors, theHierarchy);
  algos["sensor"] = new SurveyAlignmentSensor(sensors, theHierarchy);

  algos[theAlgorithm]->iterate(theIterations, theOutputFile, theBiasFlag);

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

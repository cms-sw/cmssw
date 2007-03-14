#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/SurveyAnalysis/interface/SurveyAlignmentPoints.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Alignment/SurveyAnalysis/interface/SurveyAlignmentAlgorithm.h"

SurveyAlignmentAlgorithm::SurveyAlignmentAlgorithm(const edm::ParameterSet& cfg):
  AlignmentAlgorithmBase(cfg),
  theOutfile(cfg.getParameter<std::string>("outfile")),
  theIterations(cfg.getParameter<unsigned int>("nIteration"))
{
}

void SurveyAlignmentAlgorithm::initialize(const edm::EventSetup&,
					  AlignableTracker*,
					  AlignableMuon*,
					  AlignmentParameterStore* store)
{
  const std::vector<Alignable*>& alignables = store->alignables();

  unsigned int nAlignable = alignables.size();

  std::vector<Alignable*> sensors(nAlignable, 0);

  for (unsigned int i = 0; i < nAlignable; ++i)
  {
    sensors[i] = alignables[i]->components().front();
  }

  SurveyAlignmentPoints align(sensors);

  align.iterate(theIterations, theOutfile);
}

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmPluginFactory.h"
#include "PluginManager/ModuleDef.h"

DEFINE_SEAL_MODULE();
DEFINE_SEAL_PLUGIN(AlignmentAlgorithmPluginFactory, SurveyAlignmentAlgorithm, "SurveyAlignmentAlgorithm");

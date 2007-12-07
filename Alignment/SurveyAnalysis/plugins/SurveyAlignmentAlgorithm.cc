#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/SurveyAnalysis/interface/SurveyAlignmentSensor.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Alignment/SurveyAnalysis/plugins/SurveyAlignmentAlgorithm.h"

SurveyAlignmentAlgorithm::SurveyAlignmentAlgorithm(const edm::ParameterSet& cfg):
  AlignmentAlgorithmBase(cfg),
  theOutfile(cfg.getParameter<std::string>("outfile")),
  theIterations(cfg.getParameter<unsigned int>("nIteration")),
  theLevels(cfg.getParameter< std::vector<std::string> >("levels"))
{
}

void SurveyAlignmentAlgorithm::initialize(const edm::EventSetup&,
					  AlignableTracker*,
					  AlignableMuon*,
					  AlignmentParameterStore* store)
{
  AlignableObjectId dummy;

  std::vector<Alignable::AlignableObjectIdType> levels;

  for (unsigned int l = 0; l < theLevels.size(); ++l)
  {
    levels.push_back(dummy.nameToType(theLevels[l]));
  }

  SurveyAlignmentSensor align(store->alignables(), levels);

  align.iterate(theIterations, theOutfile);
}

// Plug in to framework

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmPluginFactory.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_EDM_PLUGIN(AlignmentAlgorithmPluginFactory, SurveyAlignmentAlgorithm, "SurveyAlignmentAlgorithm");

#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"  
#include "Alignment/CommonAlignment/interface/SurveyResidual.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Alignment/CommonAlignmentMonitor/plugins/AlignmentMonitorSurvey.h"

AlignmentMonitorSurvey::AlignmentMonitorSurvey(const edm::ParameterSet& cfg)
  :AlignmentMonitorBase(cfg, "AlignmentMonitorSurvey")
{
  const std::vector<std::string>& levels = cfg.getUntrackedParameter< std::vector<std::string> >("surveyResiduals");

  for (unsigned int l = 0; l < levels.size(); ++l)
  {
    theLevels.push_back(AlignableObjectId::stringToId(levels[l]) );
  }
}

void AlignmentMonitorSurvey::book()
{
  align::ID id;
  align::StructureType level;

  double par[6]; // survey residual

  TTree* tree = directory("/iterN/")->make<TTree>("survey", "");

  tree->Branch("id"   , &id   , "id/i");
  tree->Branch("level", &level, "level/I");
  tree->Branch("par"  , &par  , "par[6]/D");

  const align::Alignables& all = pStore()->alignables();

  const unsigned int nAlignable = all.size();

  for (unsigned int i = 0; i < nAlignable; ++i)
  {
    const Alignable* ali = all[i];

    id = ali->id();

    for (unsigned int l = 0; l < theLevels.size(); ++l)
    {
      level = theLevels[l];

      SurveyResidual resid(*ali, level, true);
      AlgebraicVector resParams = resid.sensorResidual();
			
      par[0] = resParams[0];
      par[1] = resParams[1];
      par[2] = resParams[2];
      par[3] = resParams[3];
      par[4] = resParams[4];
      par[5] = resParams[5];
		
      tree->Fill();
    }
  }
}

#include "Alignment/CommonAlignmentMonitor/interface/AlignmentMonitorPluginFactory.h"
DEFINE_EDM_PLUGIN(AlignmentMonitorPluginFactory, AlignmentMonitorSurvey, "AlignmentMonitorSurvey");

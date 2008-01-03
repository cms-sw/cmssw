#include "Alignment/CommonAlignment/interface/SurveyResidual.h"

#include "Alignment/CommonAlignmentMonitor/plugins/AlignmentMonitorSurvey.h"

AlignmentMonitorSurvey::AlignmentMonitorSurvey(const edm::ParameterSet& cfg)
  :AlignmentMonitorBase(cfg)
{
}

void AlignmentMonitorSurvey::book()
{
  align::ID id;
  align::StructureType level;

  double par[6]; // survey residual

  TTree* tree = static_cast<TTree*>(add("/iterN/", new TTree("survey", "")));

  tree->Branch("id"   , &id   , "id/i");
  tree->Branch("level", &level, "level/I");
  tree->Branch("par"  , &par  , "par[6]/D");

  const align::Alignables& all = pStore()->alignables();

  const unsigned int nAlignable = all.size();

  for (unsigned int i = 0; i < nAlignable; ++i)
  {
    const Alignable* ali = all[i];

    id = ali->id();

    for ( const Alignable* mom = ali; mom->mother() != 0; mom = mom->mother() )
    {
      level = mom->alignableObjectId();	

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

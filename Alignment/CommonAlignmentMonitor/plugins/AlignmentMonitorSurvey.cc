#include "Alignment/CommonAlignment/interface/SurveyResidual.h"

#include "Alignment/CommonAlignmentMonitor/plugins/AlignmentMonitorSurvey.h"

AlignmentMonitorSurvey::AlignmentMonitorSurvey(const edm::ParameterSet& cfg)
  :AlignmentMonitorBase(cfg)
{
}

void AlignmentMonitorSurvey::book()
{
  m_tree = static_cast<TTree*>(add("/iterN/", new TTree("survey", "")));

  m_tree->Branch("id"   , &m_ID   , "id/i");
  m_tree->Branch("level", &m_level, "level/I");
  m_tree->Branch("par"  , &m_par  , "par[6]/D");
}

void AlignmentMonitorSurvey::afterAlignment(const edm::EventSetup&)
{
  const align::Alignables& all = pStore()->alignables();

  const unsigned int nAlignable = all.size();

  for (unsigned int i = 0; i < nAlignable; ++i)
  {
    const Alignable* ali = all[i];

    m_ID = ali->id();

    for ( const Alignable* mom = ali; mom->mother() != 0; mom = mom->mother() )
    {
      m_level = mom->alignableObjectId();	

      SurveyResidual resid(*ali, m_level, true);
      AlgebraicVector resParams = resid.sensorResidual();
			
      m_par[0] = resParams[0];
      m_par[1] = resParams[1];
      m_par[2] = resParams[2];
      m_par[3] = resParams[3];
      m_par[4] = resParams[4];
      m_par[5] = resParams[5];
		
      m_tree->Fill();
    }
  }
}

#include "Alignment/CommonAlignmentMonitor/interface/AlignmentMonitorPluginFactory.h"
DEFINE_EDM_PLUGIN(AlignmentMonitorPluginFactory, AlignmentMonitorSurvey, "AlignmentMonitorSurvey");

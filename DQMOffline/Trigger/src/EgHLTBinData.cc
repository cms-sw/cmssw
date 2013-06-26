#include "DQMOffline/Trigger/interface/EgHLTBinData.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"


void egHLT::BinData::setup(const edm::ParameterSet& conf)
{
  energy.setup(conf.getParameter<edm::ParameterSet>("energy"));
  et.setup(conf.getParameter<edm::ParameterSet>("et"));
  etHigh.setup(conf.getParameter<edm::ParameterSet>("etHigh"));
  eta.setup(conf.getParameter<edm::ParameterSet>("eta"));
  phi.setup(conf.getParameter<edm::ParameterSet>("phi")); 
  charge.setup(conf.getParameter<edm::ParameterSet>("charge")); 
  hOverE.setup(conf.getParameter<edm::ParameterSet>("hOverE")); 
  dPhiIn.setup(conf.getParameter<edm::ParameterSet>("dPhiIn")); 
  dEtaIn.setup(conf.getParameter<edm::ParameterSet>("dEtaIn")); 
  sigEtaEta.setup(conf.getParameter<edm::ParameterSet>("sigEtaEta")); 
  e2x5.setup(conf.getParameter<edm::ParameterSet>("e2x5")); 
  e1x5.setup(conf.getParameter<edm::ParameterSet>("e1x5")); 
  //----Morse----
  //r9.setup(conf.getParameter<edm::ParameterSet>("r9")); 
  minr9.setup(conf.getParameter<edm::ParameterSet>("minr9")); 
  maxr9.setup(conf.getParameter<edm::ParameterSet>("maxr9")); 
  HLTenergy.setup(conf.getParameter<edm::ParameterSet>("HLTenergy")); 
  HLTeta.setup(conf.getParameter<edm::ParameterSet>("HLTeta")); 
  HLTphi.setup(conf.getParameter<edm::ParameterSet>("HLTphi")); 
  deltaE.setup(conf.getParameter<edm::ParameterSet>("deltaE"));
  //--------
  isolEm.setup(conf.getParameter<edm::ParameterSet>("isolEm")); 
  isolHad.setup(conf.getParameter<edm::ParameterSet>("isolHad")); 
  isolPtTrks.setup(conf.getParameter<edm::ParameterSet>("isolPtTrks"));
  isolNrTrks.setup(conf.getParameter<edm::ParameterSet>("isolNrTrks")); 
  mass.setup(conf.getParameter<edm::ParameterSet>("mass"));
  massHigh.setup(conf.getParameter<edm::ParameterSet>("massHigh"));
  eOverP.setup(conf.getParameter<edm::ParameterSet>("eOverP"));
  invEInvP.setup(conf.getParameter<edm::ParameterSet>("invEInvP")); 
  etaVsPhi.setup(conf.getParameter<edm::ParameterSet>("etaVsPhi"));
  nVertex.setup(conf.getParameter<edm::ParameterSet>("nVertex"));
  
}


void egHLT::BinData::Data1D::setup(const edm::ParameterSet& conf)
{
  nr = conf.getParameter<int>("nr");
  min = conf.getParameter<double>("min");
  max = conf.getParameter<double>("max");
}

void egHLT::BinData::Data2D::setup(const edm::ParameterSet& conf)
{
  nrX = conf.getParameter<int>("nrX");
  xMin = conf.getParameter<double>("yMin");
  xMax = conf.getParameter<double>("yMax");  
  nrY = conf.getParameter<int>("nrY");
  yMin = conf.getParameter<double>("yMin");
  yMax = conf.getParameter<double>("yMax");
}

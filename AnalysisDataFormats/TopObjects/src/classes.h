#include "AnalysisDataFormats/TopObjects/interface/TopElectronObject.h"
#include "AnalysisDataFormats/TopObjects/interface/TopMuonObject.h"
#include "AnalysisDataFormats/TopObjects/interface/TopJetObject.h"
#include "AnalysisDataFormats/TopObjects/interface/TopMETObject.h"
#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"
#include "AnalysisDataFormats/TopObjects/interface/TtSemiEvtSolution.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include <vector>

namespace { namespace {
  //say which template classes should have dictionaries
  edm::Wrapper<TopElectronObject> dummy1;
  edm::Wrapper<TopMuonObject> dummy2;
  edm::Wrapper<TopJetObject> dummy3;
  edm::Wrapper<TopMETObject> dummy4;
  edm::Wrapper<std::vector<TopElectronObject> > dummy5;
  edm::Wrapper<std::vector<TopMuonObject> > dummy6;
  edm::Wrapper<std::vector<TopJetObject> > dummy7;
  edm::Wrapper<std::vector<TopMETObject> > dummy8;
  edm::Wrapper<TtGenEvent> dummy9;
  edm::Wrapper<TtSemiEvtSolution> dummy10;
  edm::Wrapper<std::vector<TtSemiEvtSolution> > dummy11;
  edm::Wrapper<TopElectron> dummy12;
  edm::Wrapper<TopMuon> dummy13;
  edm::Wrapper<TopMET> dummy14;
  edm::Wrapper<TopJet> dummy15;
  edm::Wrapper<TopParticle> dummy16;
  edm::Wrapper<std::vector<TopElectron> > dummy17;
  edm::Wrapper<std::vector<TopMuon> > dummy18;
  edm::Wrapper<std::vector<TopMET> > dummy19;
  edm::Wrapper<std::vector<TopJet> > dummy20;
  edm::Wrapper<std::vector<TopParticle> > dummy21;
} }


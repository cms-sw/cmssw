#include "AnalysisDataFormats/TopObjects/interface/TopElectronObject.h"
#include "AnalysisDataFormats/TopObjects/interface/TopMuonObject.h"
#include "AnalysisDataFormats/TopObjects/interface/TopJetObject.h"
#include "AnalysisDataFormats/TopObjects/interface/TopMETObject.h"
#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"
#include "AnalysisDataFormats/TopObjects/interface/TtSemiEvtSolution.h"

#include "DataFormats/Common/interface/Wrapper.h"

#include <vector>


namespace {
  namespace {

    edm::Wrapper<std::vector<std::pair<double, double> > > dummy0;

    edm::Wrapper<TopElectron>                              dummy10;
    edm::Wrapper<TopMuon>                                  dummy11;
    edm::Wrapper<TopJet>                                   dummy13;
    edm::Wrapper<TopMET>                                   dummy12;
    edm::Wrapper<TopParticle>                              dummy14;
    edm::Wrapper<TopObject<ElectronType> >                 dummy15;
    edm::Wrapper<TopObject<MuonType> >                     dummy16;
    edm::Wrapper<TopElectronObject>                        dummy17;
    edm::Wrapper<TopMuonObject>                            dummy18;
    edm::Wrapper<TopJetObject>                             dummy19;
    edm::Wrapper<TopMETObject>                             dummy110;

    edm::Wrapper<std::vector<TopElectron> >                dummy20;
    edm::Wrapper<std::vector<TopMuon> >                    dummy21;
    edm::Wrapper<std::vector<TopJet> >                     dummy23;
    edm::Wrapper<std::vector<TopMET> >                     dummy22;
    edm::Wrapper<std::vector<TopParticle> >                dummy24;
    edm::Wrapper<std::vector<TopObject<ElectronType> > >   dummy25;
    edm::Wrapper<std::vector<TopObject<MuonType> > >       dummy26;
    edm::Wrapper<std::vector<TopElectronObject> >          dummy27;
    edm::Wrapper<std::vector<TopMuonObject> >              dummy28;
    edm::Wrapper<std::vector<TopJetObject> >               dummy29;
    edm::Wrapper<std::vector<TopMETObject> >               dummy210;

    edm::Wrapper<TtGenEvent>                               dummy30;
    edm::Wrapper<TtSemiEvtSolution>                        dummy31;
    edm::Wrapper<std::vector<TtSemiEvtSolution> >          dummy32;

  }
}

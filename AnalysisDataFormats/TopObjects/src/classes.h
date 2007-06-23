#include "AnalysisDataFormats/TopObjects/interface/TopLepton.h"
#include "AnalysisDataFormats/TopObjects/interface/TopJet.h"
#include "AnalysisDataFormats/TopObjects/interface/TopMET.h"
#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"
#include "AnalysisDataFormats/TopObjects/interface/StGenEvent.h"
#include "AnalysisDataFormats/TopObjects/interface/TtSemiEvtSolution.h"
#include "AnalysisDataFormats/TopObjects/interface/TtDilepEvtSolution.h"
#include "AnalysisDataFormats/TopObjects/interface/StEvtSolution.h"
#include "AnalysisDataFormats/TopObjects/interface/TtSemiMassSolution.h"

#include "DataFormats/Common/interface/Wrapper.h"


namespace {
  namespace {

    edm::Wrapper<std::vector<std::pair<double, double> > >               dummy0;
    edm::Wrapper<std::pair<unsigned int, std::vector<unsigned int> > >   dummy1;
    edm::Wrapper<std::vector<std::pair<std::string, double> > >          dummy2;
    edm::Wrapper<std::vector<std::pair<std::string, reco::JetTagRef> > > dummy3;
    edm::Wrapper<std::pair<std::string, reco::JetTagRef> >               dummy4;
    edm::Wrapper<std::vector<std::pair<unsigned int, double> > >         dummy5;
    edm::Wrapper<std::pair<unsigned int, double> >		         dummy6;

    edm::Wrapper<TopElectron>                                          dummy10; 
    edm::Wrapper<TopMuon>                                              dummy11; 
    edm::Wrapper<TopJet>                                               dummy12;
    edm::Wrapper<TopMET>                                               dummy13;
    edm::Wrapper<TopParticle>                                          dummy14;
    edm::Wrapper<TopObject<TopElectronType> >                          dummy15;
    edm::Wrapper<TopObject<TopMuonType> >                              dummy16;
    edm::Wrapper<TopObject<TopJetType> >                               dummy17;
    edm::Wrapper<TopObject<TopMETType> >                               dummy18;
    edm::Wrapper<TopObject<TopParticleType> >                          dummy19;

    edm::Wrapper<std::vector<TopElectron> >                            dummy30;
    edm::Wrapper<std::vector<TopMuon> >                                dummy31;
    edm::Wrapper<std::vector<TopJet> >                                 dummy32;
    edm::Wrapper<std::vector<TopMET> >                                 dummy33;
    edm::Wrapper<std::vector<TopParticle> >                            dummy34;
    edm::Wrapper<std::vector<TopObject<TopElectronType> > >            dummy35;
    edm::Wrapper<std::vector<TopObject<TopMuonType> > >                dummy36;
    edm::Wrapper<std::vector<TopObject<TopJetType> > >                 dummy37;
    edm::Wrapper<std::vector<TopObject<TopMETType> > >                 dummy38;
    edm::Wrapper<std::vector<TopObject<TopParticleType> > >            dummy39;

    edm::Wrapper<TtGenEvent>                                           dummy50;
    edm::Wrapper<StGenEvent>                                           dummy51;
    edm::Wrapper<TtSemiEvtSolution>                                    dummy52;
    edm::Wrapper<std::vector<TtSemiEvtSolution> >                      dummy53;
    edm::Wrapper<StEvtSolution>                                        dummy54;
    edm::Wrapper<std::vector<StEvtSolution> >                          dummy55;
    edm::Wrapper<TtSemiMassSolution>              	               dummy56;
    edm::Wrapper<std::vector<TtDilepEvtSolution> >                     dummy57;
    edm::Wrapper<StEvtSolution>                                        dummy58;

  }
}

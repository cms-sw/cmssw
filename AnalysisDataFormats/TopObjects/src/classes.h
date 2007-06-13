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

#include <vector>


namespace {
  namespace {

    edm::Wrapper<std::vector<std::pair<double, double> > >             dummy0;
    edm::Wrapper<std::pair<unsigned int, std::vector<unsigned int> > > dummy1;
    edm::Wrapper<std::vector<std::pair<string, double> > >             dummy2;
    edm::Wrapper<std::vector<std::pair<string, JetTagRef> > >          dummy3;
    edm::Wrapper<std::pair<string, JetTagRef> >                        dummy4;
 
    edm::Wrapper<TopElectron>                                          dummy8; 
    edm::Wrapper<TopMuon>                                              dummy9; 
    edm::Wrapper<TopJet>                                               dummy10;
    edm::Wrapper<TopMET>                                               dummy11;
    edm::Wrapper<TopParticle>                                          dummy12;
    edm::Wrapper<TopObject<ElectronType> >                             dummy13;
    edm::Wrapper<TopObject<MuonType> >                                 dummy14;
    edm::Wrapper<TopObject<JetType> >                                  dummy15;
    edm::Wrapper<TopObject<METType> >                                  dummy16;

    edm::Wrapper<std::vector<TopElectron> >                            dummy17;
    edm::Wrapper<std::vector<TopMuon> >                                dummy18;
    edm::Wrapper<std::vector<TopJet> >                                 dummy19;
    edm::Wrapper<std::vector<TopMET> >                                 dummy20;
    edm::Wrapper<std::vector<TopParticle> >                            dummy21;
    edm::Wrapper<std::vector<TopObject<ElectronType> > >               dummy22;
    edm::Wrapper<std::vector<TopObject<MuonType> > >                   dummy23;
    edm::Wrapper<std::vector<TopObject<JetType> > >                    dummy24;
    edm::Wrapper<std::vector<TopObject<METType> > >                    dummy25;

    edm::Wrapper<TtGenEvent>                                           dummy26;
    edm::Wrapper<StGenEvent>                                           dummy27;
    edm::Wrapper<TtSemiEvtSolution>                                    dummy28;
    edm::Wrapper<std::vector<TtSemiEvtSolution> >                      dummy29;
    edm::Wrapper<StEvtSolution>                                        dummy30;
    edm::Wrapper<std::vector<StEvtSolution> >                          dummy31;
    edm::Wrapper<TtSemiMassSolution>              	               dummy32;
    edm::Wrapper<std::vector<TtDilepEvtSolution> >                     dummy33;
    edm::Wrapper<StEvtSolution>                                        dummy34;

  }
}

#include "DQM/PhysicsHWW/interface/monitor.h"


EventMonitor::Entry::Entry()
{
  for (unsigned int i=0; i<5; ++i){
    nevt[i] = 0;
  }
}

void EventMonitor::hypo_monitor::count(HypothesisType type, const char* name, double weight)
{
  std::vector<EventMonitor::Entry>::iterator itr = counters.begin();
  while (itr != counters.end() && itr->name != name) itr++;
  EventMonitor::Entry* entry(0);
  if ( itr == counters.end() ){
    counters.push_back(Entry());
    entry = &counters.back();
    entry->name = name;
  } else {
    entry = &*itr;
  }
  entry->nevt[type]++;
  entry->nevt[ALL]++;
}


EventMonitor::EventMonitor()
{
  //Constructor sets order of selections in the monitor.
  //This ensures that the selections in the dqm histograms 
  //will always be in the same order.
  std::vector<HypothesisType> hypType;
  hypType.push_back(MM);
  hypType.push_back(EE);
  hypType.push_back(EM);
  hypType.push_back(ME);

  for(unsigned int hypIdx = 0; hypIdx < hypType.size(); hypIdx++){

    monitor.count(hypType.at(hypIdx), "total events"                              , 0.0);
    monitor.count(hypType.at(hypIdx), "baseline"                                  , 0.0);
    monitor.count(hypType.at(hypIdx), "opposite sign"                             , 0.0);
    monitor.count(hypType.at(hypIdx), "full lepton selection"                     , 0.0);
    monitor.count(hypType.at(hypIdx), "extra lepton veto"                         , 0.0);
    monitor.count(hypType.at(hypIdx), "met > 20 GeV"                              , 0.0);
    monitor.count(hypType.at(hypIdx), "mll > 12 GeV"                              , 0.0);
    monitor.count(hypType.at(hypIdx), "|mll - mZ| > 15 GeV"                       , 0.0);
    monitor.count(hypType.at(hypIdx), "minMET > 20 GeV"                           , 0.0);
    monitor.count(hypType.at(hypIdx), "minMET > 40 GeV for ee/mm"                 , 0.0);
    monitor.count(hypType.at(hypIdx), "dPhiDiLepJet < 165 dg for ee/mm"           , 0.0);
    monitor.count(hypType.at(hypIdx), "SoftMuons==0"                              , 0.0);
    monitor.count(hypType.at(hypIdx), "top veto"                                  , 0.0);
    monitor.count(hypType.at(hypIdx), "ptll > 45 GeV"                             , 0.0);
    monitor.count(hypType.at(hypIdx), "njets == 0"                                , 0.0);
    monitor.count(hypType.at(hypIdx), "max(lep1.pt(),lep2.pt())>30"               , 0.0);
    monitor.count(hypType.at(hypIdx), "min(lep1.pt(),lep2.pt())>25"               , 0.0);
    monitor.count(hypType.at(hypIdx), "njets == 1"                                , 0.0);
    monitor.count(hypType.at(hypIdx), "njets == 2 or 3"                           , 0.0);
    monitor.count(hypType.at(hypIdx), "abs(jet1.eta())<4.7 && abs(jet2.eta())<4.7", 0.0);
    monitor.count(hypType.at(hypIdx), "no central jets"                           , 0.0);

  }
}

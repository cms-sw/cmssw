#include "DQM/PhysicsHWW/interface/monitor.h"
#include "FWCore/Utilities/interface/Exception.h"

EventMonitor::EventMonitor(DQMStore::IBooker& iBooker)
{
  auto addFunc = [&](std::string name) {
    auto bin = binMap_.find(name);
    if (bin == binMap_.end()) {
      int index = binMap_.size();
      binMap_[name] = index + 1;
    } else {
      throw cms::Exception("counterAlreadyExists") << "Name: " << name;
    }
  };

  addFunc("total events"                              );
  addFunc("baseline"                                  );
  addFunc("opposite sign"                             );
  addFunc("full lepton selection"                     );
  addFunc("extra lepton veto"                         );
  addFunc("met > 20 GeV"                              );
  addFunc("mll > 12 GeV"                              );
  addFunc("|mll - mZ| > 15 GeV"                       );
  addFunc("minMET > 20 GeV"                           );
  addFunc("minMET > 40 GeV for ee/mm"                 );
  addFunc("dPhiDiLepJet < 165 dg for ee/mm"           );
  addFunc("SoftMuons==0"                              );
  addFunc("top veto"                                  );
  addFunc("ptll > 45 GeV"                             );
  addFunc("njets == 0"                                );
  addFunc("max(lep1.pt(),lep2.pt())>30"               );
  addFunc("min(lep1.pt(),lep2.pt())>25"               );
  addFunc("njets == 1"                                );
  addFunc("njets == 2 or 3"                           );
  addFunc("abs(jet1.eta())<4.7 && abs(jet2.eta())<4.7");
  addFunc("no central jets"                           );

  int maxBin = binMap_.size();
  iBooker.setCurrentFolder("PhysicsHWW");
  cutflowHist_[0] = iBooker.book1D("cutflow_mm", "HWW cutflow mm", maxBin, 0, maxBin);	
  cutflowHist_[1] = iBooker.book1D("cutflow_ee", "HWW cutflow ee", maxBin, 0, maxBin);	
  cutflowHist_[2] = iBooker.book1D("cutflow_em", "HWW cutflow em", maxBin, 0, maxBin);	
  cutflowHist_[3] = iBooker.book1D("cutflow_me", "HWW cutflow me", maxBin, 0, maxBin);

  for (auto it = binMap_.begin(); it != binMap_.end(); ++it) {
    for (int i = 0; i < 4; ++i) {
      cutflowHist_[i]->setBinContent(it->second, 0);
      cutflowHist_[i]->setBinLabel(it->second, it->first.c_str(), 1);
    }
  }
}

void EventMonitor::count(HypothesisType type, const char* name, double weight) {
  auto bin = binMap_.find(name);
  if (bin != binMap_.end()) {
    cutflowHist_[type]->Fill(bin->second);
  } else {
    throw cms::Exception("counterNotFound") << "Name: " << name;
  }
}

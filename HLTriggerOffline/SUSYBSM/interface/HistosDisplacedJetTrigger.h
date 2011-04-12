#ifndef HistosDisplacedJetTrigger_H
#define HistosDisplacedJetTrigger_H

// Defines histograms used by AnalyseDisplacedJetTrigger class

#include "DQMServices/Core/interface/MonitorElement.h"

class HistosDisplacedJetTrigger {

public:

  // 
  // Histograms of event related quantities
  //

  // Number of primary vertices in event
  MonitorElement* nPV_;
  // Primary vertex z-position
  MonitorElement* PVz_;
  // Ditto if displaced jet trigger fired.
  MonitorElement* nPVPassed_;
  MonitorElement* PVzPassed_;

  //
  // Histograms of jet related quantities
  //

  // True production radius of recoJet (if MC truth available)
  MonitorElement* trueJetProdRadius_;
  // True number of displaced jets per event (if MC truth available)
  MonitorElement* trueNumDispJets_;
  // Histograms of offline recoJets
  MonitorElement* recoJetNpromptTk_;
  MonitorElement* recoJetPt_;
  MonitorElement* recoJetEta_;
  MonitorElement* recoJetEMfraction_;
  MonitorElement* recoJetHPDfraction_;
  MonitorElement* recoJetN90_;

  // Ditto, but only if recoJet is matched to a trigJet found by displaced jet trigger.
  MonitorElement* trueJetProdRadiusMatched_;
  MonitorElement* trueNumDispJetsMatched_;
  MonitorElement* recoJetNpromptTkMatched_;
  MonitorElement* recoJetPtMatched_;
  MonitorElement* recoJetEtaMatched_;
  MonitorElement* recoJetEMfractionMatched_;
  MonitorElement* recoJetHPDfractionMatched_;
  MonitorElement* recoJetN90Matched_;
  MonitorElement* recoJetPVzMatched_;

  // Sundry jet related histograms
  MonitorElement* trigJetVsRecoJetPt_;
};
#endif

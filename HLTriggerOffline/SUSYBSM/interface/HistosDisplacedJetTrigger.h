#ifndef HistosDisplacedJetTrigger_H
#define HistosDisplacedJetTrigger_H

// Defines histograms used by AnalyseDisplacedJetTrigger class

#include "DQMServices/Core/interface/MonitorElement.h"

class HistosDisplacedJetTrigger {

public:

  // True production radius of recoJet (if MC truth available)
  MonitorElement* trueJetProdRadius_;
  // Histograms of offline recoJets
  MonitorElement* recoJetNpromptTk_;
  MonitorElement* recoJetPt_;
  MonitorElement* recoJetEta_;
  MonitorElement* recoJetEMfraction_;
  MonitorElement* recoJetHPDfraction_;
  MonitorElement* recoJetN90_;

  // Ditto, but only if recoJet is matched to a trigJet found by displaced jet trigger.
  MonitorElement* trueJetProdRadiusMatched_;
  MonitorElement* recoJetNpromptTkMatched_;
  MonitorElement* recoJetPtMatched_;
  MonitorElement* recoJetEtaMatched_;
  MonitorElement* recoJetEMfractionMatched_;
  MonitorElement* recoJetHPDfractionMatched_;
  MonitorElement* recoJetN90Matched_;

  // Sundry
  MonitorElement* trigJetVsRecoJetPt_;
};
#endif

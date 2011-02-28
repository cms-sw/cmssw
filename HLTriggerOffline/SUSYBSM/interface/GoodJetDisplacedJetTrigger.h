#ifndef GoodJetDisplacedJetTrigger_H
#define GoodJetDisplacedJetTrigger_H

// Defines good jets for use by AnalyseDisplacedJetTrigger class

#include "DataFormats/PatCandidates/interface/Jet.h"

class GoodJetDisplacedJetTrigger {

public:

  GoodJetDisplacedJetTrigger(const pat::JetRef& patJet) {

  // Select good quality jets of high Pt, suitable for use in Exotica Displaced Jet Search.
  // The constructor determines which of the cuts this jet passes.

  const double nPromptTk_Max = 2.5; // max. number of allowed prompt tracks in jet
  const double pt_Min = 80; // Soft jets often have few tracks by chance
  const double eta_Max = 2.0; // Jets outside Tracker acceptance have few tracks ...
  const double minJetEM = 0.01; // Official jet cut. (Does kill exotics decaying outside tracker, but who cares ...)
  const double maxJetEM = 0.99; // To reject electrons
  const int minJetN90 = 3; // To reject leptons (official cut only 1)
  const double maxHPD = 0.95; // To reject HCAL noise

  passNpromptTk = patJet->bDiscriminator("displacedJetTags") < nPromptTk_Max;
  passPt  = patJet->pt() > pt_Min;
  passEta = fabs(patJet->eta()) < eta_Max;
  passJetID = (patJet->emEnergyFraction() > minJetEM && patJet->emEnergyFraction() < maxJetEM && 
               patJet->jetID().n90Hits >= minJetN90 && patJet->jetID().fHPD < maxHPD);
}

  // Decide if this jet passed all cuts. 
  bool ok() {return passNpromptTk && passPt && passEta && passJetID;}

public:

  bool passNpromptTk;
  bool passPt;
  bool passEta;
  bool passJetID; 
};
#endif

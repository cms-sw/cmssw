#ifndef GoodJetDisplacedJetTrigger_H
#define GoodJetDisplacedJetTrigger_H

// Defines good jets for use by AnalyseDisplacedJetTrigger class

#include "DataFormats/PatCandidates/interface/Jet.h"

// Select good quality jets of high Pt, suitable for use in Exotica Displaced Jet Search.
// Call constructor for each jet you are interested in and then function OK() will
// tell you if it passed all cuts. And the public boolean data members will tell you
// which individual cuts it passed.

// The default cuts are specified at the end of this file.
// They can be overridden by the user with the setCuts function.
// The kinematic cuts below should be tight enough that these jets are likely to pass
// the kinematic requirements of the displaced jet trigger.
// The prompt track cut should correspond to that used by the trigger.

class GoodJetDisplacedJetTrigger {

public:

  GoodJetDisplacedJetTrigger(const pat::JetRef& patJet) {

    passNpromptTk = patJet->bDiscriminator("displacedJetTags") < nPromptTk_Max;
    passPt  = patJet->pt() > pt_Min;
    passEta = fabs(patJet->eta()) < eta_Max;
    passJetID = (patJet->emEnergyFraction() > minJetEM && patJet->emEnergyFraction() < maxJetEM && 
                 patJet->jetID().n90Hits >= minJetN90 && patJet->jetID().fHPD < maxHPD);
  }

  // Allow user to override default cuts used to define good offline jets. 
  static void setCuts(double nPromptTkMax, double ptMin, double etaMax) {
    nPromptTk_Max = nPromptTkMax; pt_Min = ptMin; eta_Max = etaMax;
  }

 // Decide if this jet passed all cuts. 
  bool ok() {return passNpromptTk && passPt && passEta && passJetID;}

public:

  // Booleans indicating if a particular jet passed certain cuts.
  bool passNpromptTk;
  bool passPt;
  bool passEta;
  bool passJetID; 

private:

  // Cuts on defining good offline jets.

  static double nPromptTk_Max;
  static double pt_Min; 
  static double eta_Max;
  static double minJetEM;
  static double maxJetEM;
  static int    minJetN90;
  static double maxHPD;
};

// Specify default cuts to select good offline jets.
// The kinematic cuts below should be tight enough that these jets are likely to pass
// the kinematic requirements of the displaced jet trigger.
// The prompt track cut should correspond to that used by the trigger.

double GoodJetDisplacedJetTrigger::nPromptTk_Max = 2.5; // max. number of allowed prompt tracks in jet
double GoodJetDisplacedJetTrigger::pt_Min = 80; // Soft jets often have few tracks by chance
double GoodJetDisplacedJetTrigger::eta_Max = 2.0; // Jets outside Tracker acceptance have few tracks ...
double GoodJetDisplacedJetTrigger::minJetEM = 0.01; // Official jet cut. (Does kill exotics decaying outside tracker, but who cares ...)
double GoodJetDisplacedJetTrigger::maxJetEM = 0.99; // To reject electrons
int    GoodJetDisplacedJetTrigger::minJetN90 = 3; // To reject leptons (official cut only 1)
double GoodJetDisplacedJetTrigger::maxHPD = 0.95; // To reject HCAL noise
#endif

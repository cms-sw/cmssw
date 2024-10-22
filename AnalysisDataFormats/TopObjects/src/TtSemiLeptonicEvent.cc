#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "AnalysisDataFormats/TopObjects/interface/TtSemiLeptonicEvent.h"
#include "AnalysisDataFormats/TopObjects/interface/TtSemiLepEvtPartons.h"

#include "printParticle.h"
using ttevent::printParticle;

// print info via MessageLogger
void TtSemiLeptonicEvent::print(const int verbosity) const {
  if (verbosity % 10 <= 0)
    return;

  edm::LogInfo log("TtSemiLeptonicEvent");

  log << "++++++++++++++++++++++++++++++++++++++++++++++++++ \n";

  // get some information from the genEvent (if available)
  if (!genEvt_)
    log << " TtGenEvent not available! \n";
  else {
    log << " TtGenEvent says: ";
    if (!this->genEvent()->isTtBar())
      log << "Not TtBar";
    else if (this->genEvent()->isFullHadronic())
      log << "Fully Hadronic TtBar";
    else if (this->genEvent()->isFullLeptonic())
      log << "Fully Leptonic TtBar";
    else if (this->genEvent()->isSemiLeptonic()) {
      log << "Semi-leptonic TtBar, ";
      switch (this->genEvent()->semiLeptonicChannel()) {
        case WDecay::kElec:
          log << "Electron";
          break;
        case WDecay::kMuon:
          log << "Muon";
          break;
        case WDecay::kTau:
          log << "Tau";
          break;
        default:
          log << "Unknown";
          break;
      }
      log << " Channel";
    }
    log << "\n";
  }

  // get number of available hypothesis classes
  log << " Number of available event hypothesis classes: " << this->numberOfAvailableHypoClasses() << " \n";

  // create a legend for the jetLepComb
  log << " - JetLepComb: ";
  for (unsigned idx = 0; idx < 5; idx++) {
    switch (idx) {
      case TtSemiLepEvtPartons::LightQ:
        log << "LightP ";
        break;
      case TtSemiLepEvtPartons::LightQBar:
        log << "LightQ ";
        break;
      case TtSemiLepEvtPartons::HadB:
        log << " HadB  ";
        break;
      case TtSemiLepEvtPartons::LepB:
        log << " LepB  ";
        break;
      case TtSemiLepEvtPartons::Lepton:
        log << "Lepton ";
        break;
    }
  }
  log << "\n";

  // get details from the hypotheses
  typedef std::map<HypoClassKey, std::vector<HypoCombPair> >::const_iterator EventHypo;
  for (EventHypo hyp = evtHyp_.begin(); hyp != evtHyp_.end(); ++hyp) {
    HypoClassKey hypKey = (*hyp).first;
    // header for each hypothesis
    log << "-------------------------------------------------- \n";
    switch (hypKey) {
      case kGeom:
        log << " Geom";
        break;
      case kWMassMaxSumPt:
        log << " WMassMaxSumPt";
        break;
      case kMaxSumPtWMass:
        log << " MaxSumPtWMass";
        break;
      case kGenMatch:
        log << " GenMatch";
        break;
      case kMVADisc:
        log << " MVADisc";
        break;
      case kKinFit:
        log << " KinFit";
        break;
      case kKinSolution:
        log << " KinSolution not (yet) applicable to TtSemiLeptonicEvent --> skipping";
        continue;
      case kWMassDeltaTopMass:
        log << " WMassDeltaTopMass";
        break;
      case kHitFit:
        log << " HitFit";
        break;
      default:
        log << " Unknown TtEvent::HypoClassKey provided --> skipping";
        continue;
    }
    log << "-Hypothesis: \n";
    log << " * Number of real neutrino solutions: " << this->numberOfRealNeutrinoSolutions(hypKey) << "\n";
    log << " * Number of considered jets        : " << this->numberOfConsideredJets(hypKey) << "\n";
    unsigned nOfHyp = this->numberOfAvailableHypos(hypKey);
    if (nOfHyp > 1) {
      log << " * Number of stored jet combinations: " << nOfHyp << "\n";
      if (verbosity < 10)
        log << " The following was found to be the best one:\n";
    }
    // if verbosity level is smaller than 10, never show more than the best jet combination
    if (verbosity < 10)
      nOfHyp = 1;
    for (unsigned cmb = 0; cmb < nOfHyp; cmb++) {
      // check if hypothesis is valid
      if (!this->isHypoValid(hypKey, cmb))
        log << " * Not valid! \n";
      // get meta information for valid hypothesis
      else {
        // jetLepComb
        log << " * JetLepComb:";
        std::vector<int> jets = this->jetLeptonCombination(hypKey, cmb);
        for (unsigned int iJet = 0; iJet < jets.size(); iJet++) {
          log << "   " << jets[iJet] << "   ";
        }
        log << "\n";
        // specialties for some hypotheses
        switch (hypKey) {
          case kGenMatch:
            log << " * Sum(DeltaR) : " << this->genMatchSumDR(cmb) << " \n"
                << " * Sum(DeltaPt): " << this->genMatchSumPt(cmb) << " \n";
            break;
          case kMVADisc:
            log << " * Method  : " << this->mvaMethod() << " \n"
                << " * Discrim.: " << this->mvaDisc(cmb) << " \n";
            break;
          case kKinFit:
            log << " * Chi^2      : " << this->fitChi2(cmb) << " \n"
                << " * Prob(Chi^2): " << this->fitProb(cmb) << " \n";
            break;
          case kHitFit:
            log << " * Chi^2      : " << this->hitFitChi2(cmb) << " \n"
                << " * Prob(Chi^2): " << this->hitFitProb(cmb) << " \n"
                << " * Top mass   : " << this->hitFitMT(cmb) << " +/- " << this->hitFitSigMT(cmb) << " \n";
            break;
          default:
            break;
        }
        // kinematic quantities of particles (if last digit of verbosity level > 1)
        if (verbosity % 10 >= 2) {
          log << " * Candidates (pt; eta; phi; mass):\n";
          if (verbosity % 10 >= 3)
            printParticle(log, "top pair", this->topPair(hypKey, cmb));
          printParticle(log, "hadronic top", this->hadronicDecayTop(hypKey, cmb));
          printParticle(log, "hadronic W  ", this->hadronicDecayW(hypKey, cmb));
          if (verbosity % 10 >= 3) {
            printParticle(log, "hadronic b  ", this->hadronicDecayB(hypKey, cmb));
            printParticle(log, "hadronic p  ", this->hadronicDecayQuark(hypKey, cmb));
            printParticle(log, "hadronic q  ", this->hadronicDecayQuarkBar(hypKey, cmb));
          }
          printParticle(log, "leptonic top", this->leptonicDecayTop(hypKey, cmb));
          printParticle(log, "leptonic W  ", this->leptonicDecayW(hypKey, cmb));
          if (verbosity % 10 >= 3) {
            printParticle(log, "leptonic b  ", this->leptonicDecayB(hypKey, cmb));
            printParticle(log, "lepton      ", this->singleLepton(hypKey, cmb));
            printParticle(log, "neutrino    ", this->singleNeutrino(hypKey, cmb));
          }
        }
      }
    }
  }

  log << "++++++++++++++++++++++++++++++++++++++++++++++++++";
}

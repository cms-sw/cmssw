#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "AnalysisDataFormats/TopObjects/interface/TtSemiLeptonicEvent.h"
#include "AnalysisDataFormats/TopObjects/interface/TtSemiLepEvtPartons.h"

// print info via MessageLogger
void
TtSemiLeptonicEvent::print()
{
  edm::LogInfo log("TtSemiLeptonicEvent");

  log << "++++++++++++++++++++++++++++++++++++++++++++++++++ \n";

  // get some information from the genEvent (if available)
  if( !genEvt_ ) log << " TtGenEvent not available! \n";
  else {
    log << " TtGenEvent says: ";
    if( !this->genEvent()->isTtBar() )            log << "Not TtBar";
    else if( this->genEvent()->isFullHadronic() ) log << "Fully Hadronic TtBar";
    else if( this->genEvent()->isFullLeptonic() ) log << "Fully Leptonic TtBar";
    else if( this->genEvent()->isSemiLeptonic() ) {
      log << "Semi-leptonic TtBar, ";
      switch( this->genEvent()->semiLeptonicChannel() ) {
      case WDecay::kElec : log << "Electron"; break;
      case WDecay::kMuon : log << "Muon"    ; break;
      case WDecay::kTau  : log << "Tau"     ; break;
      default            : log << "Unknown" ; break;
      }
      log << " Channel";
    }
    log << "\n";
  }

  // get number of available hypothesis classes
  log << " Number of available event hypothesis classes: " << this->numberOfAvailableHypoClasses() << " \n";

  // create a legend for the jetLepComb
  log << " - JetLepComb: ";
  for(unsigned idx = 0; idx < 5; idx++) {
    switch(idx) {
    case TtSemiLepEvtPartons::LightQ    : log << "LightP "; break;
    case TtSemiLepEvtPartons::LightQBar : log << "LightQ "; break;
    case TtSemiLepEvtPartons::HadB      : log << " HadB  "; break;
    case TtSemiLepEvtPartons::LepB      : log << " LepB  "; break;
    case TtSemiLepEvtPartons::Lepton    : log << "Lepton "; break;
    }
  }
  log << "\n";

  // get details from the hypotheses
  typedef std::map<HypoClassKey, std::vector<HypoCombPair> >::const_iterator EventHypo;
  for(EventHypo hyp = evtHyp_.begin(); hyp != evtHyp_.end(); ++hyp) {
    HypoClassKey hypKey = (*hyp).first;
    // header for each hypothesis
    log << "-------------------------------------------------- \n";
    switch(hypKey) {
    case kGeom          : log << " Geom"         ; break;
    case kWMassMaxSumPt : log << " WMassMaxSumPt"; break;
    case kMaxSumPtWMass : log << " MaxSumPtWMass"; break;
    case kGenMatch      : log << " GenMatch"     ; break;
    case kMVADisc       : log << " MVADisc"      ; break;
    case kKinFit        : log << " KinFit"       ; break;
    default             : log << " Unknown";
    }
    log << "-Hypothesis: \n";
    if( this->numberOfAvailableHypos(hypKey) > 1 ) {
      log << " * Number of available jet combinations: "
	  << this->numberOfAvailableHypos(hypKey) << " \n"
	  << " The following was found to be the best one: \n";
    }
    // check if hypothesis is valid
    if( !this->isHypoValid( hypKey ) )
      log << " * Not valid! \n";
    // get meta information for valid hypothesis
    else {
      // jetLepComb
      log << " * JetLepComb:";
      std::vector<int> jets = this->jetLeptonCombination( hypKey );
      for(unsigned int iJet = 0; iJet < jets.size(); iJet++) {
	log << "   " << jets[iJet] << "   ";
      }
      log << "\n";
      // specialties for some hypotheses
      switch(hypKey) {
      case kGenMatch : log << " * Sum(DeltaR) : " << this->genMatchSumDR() << " \n"
			   << " * Sum(DeltaPt): " << this->genMatchSumPt() << " \n"; break;
      case kMVADisc  : log << " * Method  : "     << this->mvaMethod()     << " \n"
			   << " * Discrim.: "     << this->mvaDisc()       << " \n"; break;
      case kKinFit   : log << " * Chi^2      : "  << this->fitChi2()       << " \n"
			   << " * Prob(Chi^2): "  << this->fitProb()       << " \n"; break;
      default        : break;
      }
    }
  }

  log << "++++++++++++++++++++++++++++++++++++++++++++++++++";  
}

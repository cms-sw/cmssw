#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "AnalysisDataFormats/TopObjects/interface/TtFullHadronicEvent.h"
#include "AnalysisDataFormats/TopObjects/interface/TtFullHadEvtPartons.h"

// print info via MessageLogger
void
TtFullHadronicEvent::print()
{
  edm::LogInfo log("TtFullHadronicEvent");

  log << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ \n";

  // get some information from the genEvent
  log << " TtGenEvent says: ";
  if( !this->genEvent()->isTtBar() )            log << "Not TtBar";
  else if( this->genEvent()->isFullHadronic() ) log << "Fully Hadronic TtBar";
  else if( this->genEvent()->isSemiLeptonic() ) log << "Semi-leptonic TtBar";
  else if( this->genEvent()->isFullLeptonic() ) log << "Fully Leptonic TtBar";
  log << "\n";
  
  // get number of available hypothesis classes
  log << " Number of available event hypothesis classes: " << this->numberOfAvailableHypoClasses() << " \n";

  // create a legend for the jetLepComb
  log << " - JetLepComb: ";
  for(unsigned idx = 0; idx < 6; idx++) {
    switch(idx) {
    case TtFullHadEvtPartons::LightQ    : log << "  LightQ  "; break;
    case TtFullHadEvtPartons::LightQBar : log << "LightQBar "; break;
    case TtFullHadEvtPartons::B         : log << "    B     "; break;
    case TtFullHadEvtPartons::LightP    : log << "  LightP  "; break;
    case TtFullHadEvtPartons::LightPBar : log << "LightPBar "; break;
    case TtFullHadEvtPartons::BBar      : log << "   BBar   "; break;
    }
  }
  log << "\n";

  // get details from the hypotheses
  typedef std::map<HypoClassKey, std::vector<HypoCombPair> >::const_iterator EventHypo;
  for(EventHypo hyp = evtHyp_.begin(); hyp != evtHyp_.end(); ++hyp) {
    HypoClassKey hypKey = (*hyp).first;
    // header for each hypothesis
    log << "------------------------------------------------------------------------ \n";
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
	log << "     " << jets[iJet] << "    ";
      }
      log << "\n";
      // specialties for some hypotheses
      switch(hypKey) {
      case kGenMatch : log << " * Sum(DeltaR) : " << this->genMatchSumDR()   << " \n"
			   << " * Sum(DeltaPt): " << this->genMatchSumPt()   << " \n"; break;      
      case kMVADisc  : log << " * Method  : "     << this->mvaMethod()     << " \n"
			   << " * Discrim.: "     << this->mvaDisc()       << " \n"; break;
      case kKinFit   : log << " * Chi^2      : "  << this->fitChi2()       << " \n"
			   << " * Prob(Chi^2): "  << this->fitProb()       << " \n"; break;
      default        : break;
      }
    }
  }
  log << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++";
}

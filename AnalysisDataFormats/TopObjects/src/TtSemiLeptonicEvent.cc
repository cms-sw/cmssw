#include "FWCore/MessageLogger/interface/MessageLogger.h"
//#include "TopQuarkAnalysis/TopTools/interface/TtSemiLepEvtPartons.h"
#include "AnalysisDataFormats/TopObjects/interface/TtSemiLeptonicEvent.h"

//empty constructor
TtSemiLeptonicEvent::TtSemiLeptonicEvent():
  decay_(kNone)
{
}

// find corresponding jet matches in two hypotheses
int
TtSemiLeptonicEvent::correspondingJetMatch(const HypoKey& key1, const unsigned& cmb1, const HypoKey& key2) const
{
  for(unsigned cmb2 = 0; cmb2 < this->numberOfAvailableCombs(key2); ++cmb2) {
    if( this->jetMatch(key1, cmb1) ==  this->jetMatch(key2, cmb2) )
      return cmb2;
  }
  return -1; // if no corresponding jet match was found
}

// print info via MessageLogger
void
TtSemiLeptonicEvent::print()
{
  // get some information from the genEvent
  std::string genEvtString = " TtGenEvent says: ";
  if( !this->genEvent()->isTtBar() )            genEvtString += "Not TtBar";
  else if( this->genEvent()->isFullHadronic() ) genEvtString += "Fully Hadronic TtBar";
  else if( this->genEvent()->isFullLeptonic() ) genEvtString += "Fully Leptonic TtBar";
  else if( this->genEvent()->isSemiLeptonic() ) {
    genEvtString += "Semi-leptonic TtBar, ";
    switch( this->genEvent()->semiLeptonicChannel() ) {
    case TtGenEvent::kElec : genEvtString += "Electron"; break;
    case TtGenEvent::kMuon : genEvtString += "Muon"    ; break;
    case TtGenEvent::kTau  : genEvtString += "Tau"     ; break;
    default                : genEvtString += "Unknown" ; break;
    }
    genEvtString += " Channel";
  }

  // create a legend for the jetMatch
  std::string jetString = " - JetMatch: ";
  for(unsigned idx = 0; idx < 4; idx++) {
    switch(idx) {
    case TtSemiLepEvtPartons::LightQ    : jetString += "LightP "; break;
    case TtSemiLepEvtPartons::LightQBar : jetString += "LightQ "; break;
    case TtSemiLepEvtPartons::HadB      : jetString += " HadB  "  ; break;
    case TtSemiLepEvtPartons::LepB      : jetString += " LepB  "  ; break;
    }
  }

  // use stringstream to collect information from the hypotheses for the MessageLogger
  std::stringstream hypStream;
  
  typedef std::map<HypoKey, std::vector<HypoCombPair> >::const_iterator EventHypo;
  for(EventHypo hyp = evtHyp_.begin(); hyp != evtHyp_.end(); ++hyp) {
    HypoKey hypKey = (*hyp).first;
    // header for each hypothesis
    hypStream << "--------------------------------------------- \n";
    switch(hypKey) {
    case kGeom          : hypStream << " Geom"         ; break;
    case kWMassMaxSumPt : hypStream << " WMassMaxSumPt"; break;
    case kMaxSumPtWMass : hypStream << " MaxSumPtWMass"; break;
    case kGenMatch      : hypStream << " GenMatch"     ; break;
    case kMVADisc       : hypStream << " MVADisc"      ; break;
    case kKinFit        : hypStream << " KinFit"       ; break;
    default                                  : hypStream << " Unknown";
    }
    hypStream << "-Hypothesis: \n";
    if( this->numberOfAvailableCombs(hypKey) > 1 ) {
      hypStream << " * Number of available jet combinations: "
		<< this->numberOfAvailableCombs(hypKey) << " \n"
		<< " The following was found to be the best one: \n";
    }
    // check if hypothesis is valid
    if( !this->isHypoValid( hypKey ) )
      hypStream << " * Not valid! \n";
    // get meta information for valid hypothesis
    else {
      // jetMatch
      hypStream << " * JetMatch:";
      std::vector<int> jets = this->jetMatch( hypKey );
      for(unsigned int iJet = 0; iJet < jets.size(); iJet++) {
	hypStream << "   " << jets[iJet] << "   ";
      }
      hypStream << "\n";
      // specialties for some hypotheses
      switch(hypKey) {
      case kGenMatch : hypStream << " * Sum(DeltaR) : " << this->genMatchSumDR() << " \n"
				 << " * Sum(DeltaPt): " << this->genMatchSumPt() << " \n"; break;
      case kMVADisc  : hypStream << " * Method  : "     << this->mvaMethod()     << " \n"
				 << " * Discrim.: "     << this->mvaDisc()       << " \n"; break;
      case kKinFit   : hypStream << " * Chi^2      : "  << this->fitChi2()       << " \n"
				 << " * Prob(Chi^2): "  << this->fitProb()       << " \n"; break;
      default        : break;
      }
    }
  }

  // pass information to the MessageLogger
  edm::LogInfo( "TtSemiLeptonicEvent" ) 
    << "++++++++++++++++++++++++++++++++++++++++++++++ \n"
    << genEvtString << " \n"
    << " Number of available event hypotheses: " << this->numberOfAvailableHypos() << " \n"
    << jetString << " \n"
    << hypStream.str()
    << "++++++++++++++++++++++++++++++++++++++++++++++";  
}

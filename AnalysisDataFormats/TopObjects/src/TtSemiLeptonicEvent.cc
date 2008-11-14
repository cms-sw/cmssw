#include "FWCore/MessageLogger/interface/MessageLogger.h"
//#include "TopQuarkAnalysis/TopTools/interface/TtSemiLepEvtPartons.h"
#include "AnalysisDataFormats/TopObjects/interface/TtSemiLeptonicEvent.h"

//empty constructor
TtSemiLeptonicEvent::TtSemiLeptonicEvent():
  decay_(kNone), 
  fitChi2_(-1.), 
  genMatchSumPt_(-1.), 
  genMatchSumDR_(-1.)
{
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
  
  unsigned int nHyps = this->numberOfAvailableHypos();
  for(unsigned hyp = 0; hyp < nHyps; hyp++) {
    // header for each hypothesis
    hypStream << "--------------------------------------------- \n";
    switch(hyp) {
    case TtSemiLeptonicEvent::kGeom          : hypStream << " Geom"         ; break;
    case TtSemiLeptonicEvent::kWMassMaxSumPt : hypStream << " WMassMaxSumPt"; break;
    case TtSemiLeptonicEvent::kMaxSumPtWMass : hypStream << " MaxSumPtWMass"; break;
    case TtSemiLeptonicEvent::kGenMatch      : hypStream << " GenMatch"     ; break;
    case TtSemiLeptonicEvent::kMVADisc       : hypStream << " MVADisc"      ; break;
    case TtSemiLeptonicEvent::kKinFit        : hypStream << " KinFit"       ; break;
    default                                  : hypStream << " Unknown";
    }
    hypStream << "-Hypothesis: \n";
    // check if hypothesis is valid
    if( !this->isHypoValid((TtSemiLeptonicEvent::HypoKey) hyp) )
      hypStream << " * Not valid! \n";
    // get meta information for valid hypothesis
    else {
      // jetMatch
      hypStream << " * JetMatch:";
      std::vector<int> jets = this->jetMatch( (TtSemiLeptonicEvent::HypoKey&) hyp  );
      for(unsigned int iJet = 0; iJet < jets.size(); iJet++) {
	hypStream << "   " << jets[iJet] << "   ";
      }
      hypStream << "\n";
      // specialties for some hypotheses
      switch(hyp) {
      case TtSemiLeptonicEvent::kGenMatch : hypStream << " * Sum(DeltaR) : " << this->genMatchSumDR() << " \n"
						      << " * Sum(DeltaPt): " << this->genMatchSumPt() << " \n"; break;
      case TtSemiLeptonicEvent::kMVADisc  : hypStream << " * Method  : "     << this->mvaMethod()     << " \n"
						      << " * Discrim.: "     << this->mvaDisc()       << " \n"; break;
      case TtSemiLeptonicEvent::kKinFit   : hypStream << " * Chi^2      : "  << this->fitChi2()       << " \n"
						      << " * Prob(Chi^2): "  << this->fitProb()       << " \n"; break;
      }
    }
  }

  // pass information to the MessageLogger
  edm::LogInfo( "TtSemiLeptonicEvent" ) 
    << "++++++++++++++++++++++++++++++++++++++++++++++ \n"
    << genEvtString << " \n"
    << " Number of available event hypotheses: " << nHyps << " \n"
    << jetString << " \n"
    << hypStream.str()
    << "++++++++++++++++++++++++++++++++++++++++++++++";  
}

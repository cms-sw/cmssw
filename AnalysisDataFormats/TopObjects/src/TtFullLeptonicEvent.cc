#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "AnalysisDataFormats/TopObjects/interface/TtFullLeptonicEvent.h"
#include "AnalysisDataFormats/TopObjects/interface/TtFullLepEvtPartons.h"

// print info via MessageLogger
void
TtFullLeptonicEvent::print()
{
  edm::LogInfo log("TtFullLeptonicEvent");

  log << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ \n";

  // get some information from the genEvent
  log << " TtGenEvent says: ";
  if( !this->genEvent()->isTtBar() )            log << "Not TtBar";
  else if( this->genEvent()->isFullHadronic() ) log << "Fully Hadronic TtBar";
  else if( this->genEvent()->isSemiLeptonic() ) log << "Semi-leptonic TtBar";
  else if( this->genEvent()->isFullLeptonic() ) {
    log << "Fully Leptonic TtBar, ";
    switch( this->genEvent()->fullLeptonicChannel().first ) {     
    case WDecay::kElec : log << "Electron-"; break;
    case WDecay::kMuon : log << "Muon-"    ; break;
    case WDecay::kTau  : log << "Tau-"     ; break;
    default            : log << "Unknown-" ; break;
    }
    switch( this->genEvent()->fullLeptonicChannel().second ) {     
    case WDecay::kElec : log << "Electron-"; break;
    case WDecay::kMuon : log << "Muon-"    ; break;
    case WDecay::kTau  : log << "Tau-"     ; break;
    default            : log << "Unknown-" ; break;
    }        
    log << "Channel";        
  }
  log << "\n";
  
  // get number of available hypothesis classes
  log << " Number of available event hypothesis classes: " << this->numberOfAvailableHypoClasses() << " \n";

  // create a legend for the jetLepComb
  log << " - JetLepComb: ";
  log << "  b    ";         
  log << " bbar  ";
  log << " e1(+) ";  
  log << " e2(-) ";  
  log << " mu1(+)";    
  log << " mu2(-)";      
  log << "\n";

  // get details from the hypotheses
  typedef std::map<HypoClassKey, std::vector<HypoCombPair> >::const_iterator EventHypo;
  for(EventHypo hyp = evtHyp_.begin(); hyp != evtHyp_.end(); ++hyp) {
    HypoClassKey hypKey = (*hyp).first;
    // header for each hypothesis
    log << "------------------------------------------------------------ \n";
    switch(hypKey) {
    case kGeom          : log << " Geom"         ; break;
    case kWMassMaxSumPt : log << " WMassMaxSumPt"; break;
    case kMaxSumPtWMass : log << " MaxSumPtWMass"; break;
    case kGenMatch      : log << " GenMatch"     ; break;
    case kMVADisc       : log << " MVADisc"      ; break;
    case kKinFit        : log << " KinFit"       ; break;
    case kKinSolution   : log << " KinSolution"  ; break;    
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
      case kGenMatch : log << " * Sum(DeltaR) : "     << this->genMatchSumDR()   << " \n"
			   << " * Sum(DeltaPt): "     << this->genMatchSumPt()   << " \n"; break;      
      case kKinSolution : log << " * Weight      : "  << this->solWeight()       << " \n"
			      << " * isWrongCharge: " << this->isWrongCharge()   << " \n"; break;
      default        : break;
      }
    }
  }

  log << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++";  
}

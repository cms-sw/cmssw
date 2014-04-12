#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "AnalysisDataFormats/TopObjects/interface/TtFullHadronicEvent.h"
#include "AnalysisDataFormats/TopObjects/interface/TtFullHadEvtPartons.h"

// print info via MessageLogger
void
TtFullHadronicEvent::print(const int verbosity) const
{
  if(verbosity%10<=0)
    return;

  edm::LogInfo log("TtFullHadronicEvent");

  log << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ \n";

  // get some information from the genEvent
  if( !genEvt_ ) log << " TtGenEvent not available! \n";
  else {
    log << " TtGenEvent says: ";
    if( !this->genEvent()->isTtBar() )            log << "Not TtBar";
    else if( this->genEvent()->isFullHadronic() ) log << "Fully Hadronic TtBar";
    else if( this->genEvent()->isSemiLeptonic() ) log << "Semi-leptonic TtBar";
    else if( this->genEvent()->isFullLeptonic() ) log << "Fully Leptonic TtBar";
    log << "\n";
  }
  // get number of available hypothesis classes
  log << " Number of available event hypothesis classes: " << this->numberOfAvailableHypoClasses() << " \n";

  // create a legend for the jetLepComb
  log << " - JetCombi    : ";
  for(unsigned idx = 0; idx < 6; idx++) {
    switch(idx) {
    case TtFullHadEvtPartons::LightQ    : log << "  LightQ  "; break;
    case TtFullHadEvtPartons::LightQBar : log << " LightQBar"; break;
    case TtFullHadEvtPartons::B         : log << "     B    "; break;
    case TtFullHadEvtPartons::LightP    : log << "   LightP "; break;
    case TtFullHadEvtPartons::LightPBar : log << " LightPBar"; break;
    case TtFullHadEvtPartons::BBar      : log << "    BBar  "; break;
    }
  }
  log << "\n";


  // get details from the hypotheses
  typedef std::map<HypoClassKey, std::vector<HypoCombPair> >::const_iterator EventHypo;
  for(EventHypo hyp = evtHyp_.begin(); hyp != evtHyp_.end(); ++hyp) {
    HypoClassKey hypKey = (*hyp).first;
    // header for each hypothesis
    log << "---------------------------------------------------------------------------- \n";
    switch(hypKey) {
    case kGeom              : log << " Geom not (yet) applicable to TtFullHadronicEvent --> skipping"             ; continue;
    case kWMassMaxSumPt     : log << " WMassMaxSumPt not (yet) applicable to TtFullHadronicEvent --> skipping"    ; continue;
    case kMaxSumPtWMass     : log << " MaxSumPtWMass not (yet) applicable to TtFullHadronicEvent --> skipping"    ; continue;
    case kGenMatch          : log << " GenMatch"                                                                  ; break;
    case kMVADisc           : log << " MVADisc not (yet) applicable to TtFullHadronicEvent --> skipping"          ; continue;
    case kKinFit            : log << " KinFit"                                                                    ; break;
    case kKinSolution       : log << " KinSolution not (yet) applicable to TtFullHadronicEvent --> skipping"      ; continue;
    case kWMassDeltaTopMass : log << " WMassDeltaTopMass not (yet) applicable to TtFullHadronicEvent --> skipping"; continue;
    case kHitFit            : log << " HitFit not (yet) applicable to TtFullHadronicEvent --> skipping"           ; continue;
    default                 : log << " Unknown TtEvent::HypoClassKey provided --> skipping"                       ; continue;
    }
    log << "-Hypothesis: \n";
    unsigned nOfHyp = this->numberOfAvailableHypos(hypKey);
    if( nOfHyp > 1 ) {
      log << " * Number of available jet combinations: " << nOfHyp << " \n";
      if(verbosity < 10) log << " The following was found to be the best one: \n";
    }
    // if verbosity level is smaller than 10, never show more than the best jet combination
    if(verbosity < 10)
      nOfHyp = 1;
    for(unsigned cmb=0; cmb<nOfHyp; ++cmb) {
      // check if hypothesis is valid
      if( !this->isHypoValid(hypKey, cmb) )
	log << " * Not valid! \n";
      // get meta information for valid hypothesis
      else {
	// jetLepComb
	log << " * JetCombi    :";
	std::vector<int> jets = this->jetLeptonCombination( hypKey , cmb );
	for(unsigned int iJet = 0; iJet < jets.size(); iJet++) {
	  log << "      " << jets[iJet] << "   ";
	}
	log << "\n";
	// specialties for some hypotheses
	switch(hypKey) {
	case kGenMatch : log << " * Sum(DeltaR) : " << this->genMatchSumDR(cmb) << " \n"
			     << " * Sum(DeltaPt): " << this->genMatchSumPt(cmb) << " \n"; break;
	case kKinFit   : log << " * Chi^2       : " << this->fitChi2(cmb)       << " \n"
			     << " * Prob(Chi^2) : " << this->fitProb(cmb)       << " \n"; break;
	default        : break;
	}
	// kinematic quantities of particles (if last digit of verbosity level > 1)
	if(verbosity%10 >= 2) {
	  log << " * Candidates (pt; eta; phi; mass) :\n";
	  if(verbosity%10 >= 3)
	    printParticle(log, "top pair", this->topPair(hypKey, cmb));
	  printParticle(log, "top       ", this->top  (hypKey, cmb));
	  printParticle(log, "W plus    ", this->wPlus(hypKey, cmb));
	  if(verbosity%10 >= 3) {
	    printParticle(log, "b         ", this->b        (hypKey, cmb));
	    printParticle(log, "lightQ    ", this->lightQ   (hypKey, cmb));
	    printParticle(log, "lightQBar ", this->lightQBar(hypKey, cmb));
	  }
	  printParticle(log, "topBar    ", this->topBar(hypKey, cmb));
	  printParticle(log, "W minus   ", this->wMinus(hypKey, cmb));
	  if(verbosity%10 >= 3) {
	    printParticle(log, "bBar      ", this->bBar     (hypKey, cmb));
	    printParticle(log, "lightP    ", this->lightP   (hypKey, cmb));
	    printParticle(log, "lightPBar ", this->lightPBar(hypKey, cmb));
	  }
	}
      }
    }
  }
  log << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++";
}

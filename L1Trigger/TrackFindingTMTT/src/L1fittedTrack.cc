#include "L1Trigger/TrackFindingTMTT/interface/L1fittedTrack.h"

namespace TMTT {

// Digitize track and degrade helix parameter resolution according to effect of digitisation.

void L1fittedTrack::digitizeTrack(const string& fitterName){
  if (settings_->enableDigitize()) {
    if (! digitizedTrack_ ) {
      digitizedTrack_ = true;
     
      bool consistent = this->consistentHTcell();
      bool consistentSect = this->consistentSector();
      int  mbinhelix  = int(this->getCellLocationFit().first) - floor(settings_->houghNbinsPt()/2);
      int  cbinhelix  = int(this->getCellLocationFit().second) - floor(settings_->houghNbinsPhi()/2);
      int  mBinHT     = int(this->getCellLocationHT().first) - floor(settings_->houghNbinsPt()/2);
      int  cBinHT     = int(this->getCellLocationHT().second) - floor(settings_->houghNbinsPhi()/2);

      if(matchedTP_ != nullptr){
        digitalTrack_.init(fitterName, nHelixParam_,
	 iPhiSec_, iEtaReg_, mBinHT, cBinHT, mbinhelix, cbinhelix, hitPattern_,
	 qOverPt_, d0_, phi0_,tanLambda_, z0_, chi2rphi_, chi2rz_,
	 qOverPt_bcon_, phi0_bcon_, chi2rphi_bcon_, 
	 nLayers_, consistent, consistentSect, this->accepted(),
	 matchedTP_->qOverPt(), matchedTP_->d0(), matchedTP_->phi0(), matchedTP_->tanLambda(), matchedTP_->z0(), matchedTP_->eta(), 
	 matchedTP_->index(), matchedTP_->useForAlgEff(), matchedTP_->useForEff(), matchedTP_->pdgId());
      } else {
        digitalTrack_.init(fitterName, nHelixParam_,
	 iPhiSec_, iEtaReg_, mBinHT, cBinHT, mbinhelix, cbinhelix, hitPattern_,     
	 qOverPt_, d0_, phi0_, tanLambda_, z0_, chi2rphi_, chi2rz_, 
	 qOverPt_bcon_, phi0_bcon_, chi2rphi_bcon_,
	 nLayers_, consistent, consistentSect, this->accepted(), 
	 0, 0, 0, 0, 0, 0, 
	 -1, 0, 0, 0);
      }

      // Digitize track
      digitalTrack_.makeDigitalTrack();

      // Convert digitized track params back to floating point with degraded resolution.
      qOverPt_   = digitalTrack_.qOverPt();
      if (nHelixParam_ == 5) d0_ = digitalTrack_.d0();
      phi0_      = digitalTrack_.phi0();
      z0_        = digitalTrack_.z0();
      tanLambda_ = digitalTrack_.tanLambda();
      chi2rphi_  = digitalTrack_.chisquaredRphi();
      chi2rz_    = digitalTrack_.chisquaredRz();

      // Ditto for beam-spot constrained values.
      if (nHelixParam_ == 5) {
        qOverPt_bcon_   = digitalTrack_.qOverPt_bcon();
        phi0_bcon_      = digitalTrack_.phi0_bcon();
        chi2rphi_bcon_  = digitalTrack_.chisquaredRphi_bcon();
      }

      // Recalculate consistency flag using updated helix params.
      this->setConsistentHTcell();
    }
  }
}

}

#ifndef RecoTrackerDeDx_UnbinnedFitDeDxEstimator_h
#define RecoTrackerDeDx_UnbinnedFitDeDxEstimator_h

#include "DataFormats/TrackReco/interface/DeDxHit.h"
#include "RecoTracker/DeDx/interface/BaseDeDxEstimator.h"
#include "RecoTracker/DeDx/interface/DeDxTools.h"
#include "RecoTracker/DeDx/interface/UnbinnedLikelihoodFit.h"

#include <TF1.h>

#include <iostream>
#include <vector>

class UnbinnedFitDeDxEstimator: public BaseDeDxEstimator
{
 public: 

  UnbinnedFitDeDxEstimator(const edm::ParameterSet& iConfig) {
    fitter.setFunction((f1 = new TF1("myLandau","TMath::Landau(x,[0],[1],1)",0,255)));
  }
  
  ~UnbinnedFitDeDxEstimator() override {
    // clean up everything
    delete f1;
  }
 
  std::pair<float,float> dedx(const reco::DeDxHitCollection & Hits) override{
    // if there is no hit, returns invalid.
    if(Hits.empty()) return std::make_pair(-1,-1);
    // sets the initial parameters
    f1->SetParameters(3.0 , 0.3);
    // fills a temporary array and performs the fit
    uint32_t i=0;
    for (reco::DeDxHitCollection::const_iterator hit = Hits.begin(); hit!=Hits.end(); ++hit,++i) {
      data[i] = hit->charge();
    }
    // fit !
    fitter.fit(Hits.size(),data);
    // returns the mpv and its error
    return std::make_pair(f1->GetParameter(0),f1->GetParError(0)); 
  }
 
  // ----------member data ---------------------------
  double data[50];
  TF1* f1;
  UnbinnedLikelihoodFit fitter;
 
};

#endif

#ifndef RecoTauTag_RecoTau_PFRecoTauDiscriminationByCharge_H_
#define RecoTauTag_RecoTau_PFRecoTauDiscriminationByCharge_H_



#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"

using namespace std; 
using namespace edm;
//using namespace edm::eventsetup;
using namespace reco;

class PFRecoTauDiscriminationByCharge : public EDProducer {
 public:
  explicit PFRecoTauDiscriminationByCharge(const ParameterSet& iConfig){   
    PFTauProducer_        = iConfig.getParameter<InputTag>("PFTauProducer");
    ptcut_                = iConfig.getParameter<double>("PTcut");
    minHitsLeadTk_        = iConfig.getParameter<double>("MinHitsLeadTk");
    minSigPtTkRatio_      = iConfig.getParameter<double>("MinSigPtTkRatio");
    applySigTkSumPt_      = iConfig.getParameter<bool>("ApplySigTkSum");
    produces<PFTauDiscriminator>();
  }
  ~PFRecoTauDiscriminationByCharge(){} 
  virtual void produce(Event&, const EventSetup&);
 private:
  InputTag PFTauProducer_;
  double ptcut_;
  double minSigPtTkRatio_;
  bool applySigTkSumPt_;
  double minHitsLeadTk_;
};
#endif

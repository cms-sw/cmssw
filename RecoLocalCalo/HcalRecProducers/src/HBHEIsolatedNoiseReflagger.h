#ifndef __HBHE_ISOLATED_NOISE_REFLAGGER_H__
#define __HBHE_ISOLATED_NOISE_REFLAGGER_H__

/*
Description: "Reflags" HB/HE hits based on their ECAL, HCAL, and tracking isolation.

Original Author: John Paul Chou (Brown University)
                 Thursday, September 2, 2010
*/
#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoLocalCalo/HcalRecAlgos/interface/HBHEIsolatedNoiseAlgos.h"


class HBHEIsolatedNoiseReflagger : public edm::stream::EDProducer<> {
 public:
  explicit HBHEIsolatedNoiseReflagger(const edm::ParameterSet&);
  ~HBHEIsolatedNoiseReflagger();
  
  
 private:
  virtual void produce(edm::Event&, const edm::EventSetup&) override;

  void DumpHBHEHitMap(std::vector<HBHEHitMap>& i) const;

  // parameters
  edm::EDGetTokenT<HBHERecHitCollection> tok_hbhe_;
  edm::EDGetTokenT<EcalRecHitCollection> tok_EB_;
  edm::EDGetTokenT<EcalRecHitCollection> tok_EE_;
  edm::EDGetTokenT<std::vector<reco::TrackExtrapolation> > tok_trackExt_;

  double LooseHcalIsol_;
  double LooseEcalIsol_;
  double LooseTrackIsol_;
  double TightHcalIsol_;
  double TightEcalIsol_;
  double TightTrackIsol_;
  
  double LooseRBXEne1_, LooseRBXEne2_;
  int LooseRBXHits1_, LooseRBXHits2_;
  double TightRBXEne1_, TightRBXEne2_;
  int TightRBXHits1_, TightRBXHits2_;
  double LooseHPDEne1_, LooseHPDEne2_;
  int LooseHPDHits1_, LooseHPDHits2_;
  double TightHPDEne1_, TightHPDEne2_;
  int TightHPDHits1_, TightHPDHits2_;
  double LooseDiHitEne_;
  double TightDiHitEne_;
  double LooseMonoHitEne_;
  double TightMonoHitEne_;
  
  bool debug_;

  // object validator
  ObjectValidator objvalidator_;

};

#endif

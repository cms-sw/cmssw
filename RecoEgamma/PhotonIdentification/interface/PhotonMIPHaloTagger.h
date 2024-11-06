#ifndef RecoEgamma_PhotonIdentification_PhotonMIPHaloTagger_H
#define RecoEgamma_PhotonIdentification_PhotonMIPHaloTagger_H

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include <vector>

class PhotonMIPHaloTagger {
public:
  PhotonMIPHaloTagger(const edm::ParameterSet& conf, edm::ConsumesCollector&& iC);

  reco::Photon::MIPVariables mipCalculate(const reco::Photon&, const edm::Event&, const edm::EventSetup& es) const;

private:
  //get the seed crystal index
  void getSeedHighestE(const reco::Photon& photon,
                       const edm::Event& iEvent,
                       const edm::EventSetup& iSetup,
                       edm::Handle<EcalRecHitCollection> Brechit,
                       int& seedIEta,
                       int& seedIPhi,
                       double& seedE) const;

  //get the MIP  Fit Trail results
  reco::Photon::MIPVariables getMipTrailFit(const reco::Photon& photon,
                                            const edm::Event& iEvent,
                                            const edm::EventSetup& iSetup,
                                            edm::Handle<EcalRecHitCollection> ecalhitsCollEB,
                                            double inputRangeY,
                                            double inputRangeX,
                                            double inputResWidth,
                                            double inputHaloDiscCut) const;

  const edm::EDGetToken EBecalCollection_;

  //Isolation parameters variables as input
  const double yRangeFit_;
  const double xRangeFit_;
  const double residualWidthEnergy_;
  const double haloDiscThreshold_;
};

#endif  // PhotonMIPHaloTagger_H

#ifndef PhotonIDProducer_h
#define PhotonIDProducer_h

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "RecoEgamma/PhotonIdentification/interface/CutBasedPhotonIDAlgo.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"


class PhotonIDProducer : public edm::stream::EDProducer<>
{
 public:

  explicit PhotonIDProducer(const edm::ParameterSet& conf);
  ~PhotonIDProducer() override;

  void produce(edm::Event& e, const edm::EventSetup& c) override;
   
 private:

  CutBasedPhotonIDAlgo* cutBasedAlgo_; 	   

  edm::ParameterSet conf_;
  edm::EDGetTokenT<reco::PhotonCollection> photonToken_;

  std::string photonCutBasedIDLooseEMLabel_;
  std::string photonCutBasedIDLooseLabel_;
  std::string photonCutBasedIDTightLabel_;

  bool doCutBased_;

};

#endif

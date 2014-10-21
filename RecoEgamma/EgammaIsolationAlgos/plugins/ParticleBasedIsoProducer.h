#ifndef ParticleBasedIsoProducer_h
#define ParticleBasedIsoProducer_h

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/PfBlockBasedIsolation.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

#include "DataFormats/Common/interface/ValueMap.h"


class ParticleBasedIsoProducer : public edm::stream::EDProducer<>
{
 public:

  ParticleBasedIsoProducer(const edm::ParameterSet& conf);
  ~ParticleBasedIsoProducer();
  
  virtual void beginRun (edm::Run const& r, edm::EventSetup const & es) override;
  virtual void endRun(edm::Run const&,  edm::EventSetup const&) override;
  virtual void produce(edm::Event& e, const edm::EventSetup& c);
   
 private:
 
 edm::ParameterSet conf_;
 std::string photonCollection_;
 std::string electronCollection_;

 edm::InputTag  photonProducer_; 
 edm::InputTag  photonTmpProducer_;

 edm::InputTag  electronProducer_; 
 edm::InputTag  electronTmpProducer_; 

 edm::EDGetTokenT<reco::PhotonCollection> photonProducerT_;
 edm::EDGetTokenT<reco::PhotonCollection> photonTmpProducerT_;
 edm::EDGetTokenT<reco::GsfElectronCollection> electronProducerT_;
 edm::EDGetTokenT<reco::GsfElectronCollection> electronTmpProducerT_;
 edm::EDGetTokenT<reco::PFCandidateCollection> pfEgammaCandidates_;
 edm::EDGetTokenT<reco::PFCandidateCollection> pfCandidates_;
 edm::EDGetTokenT<edm::ValueMap<reco::PhotonRef> > valMapPFCandToPhoton_;
 edm::EDGetTokenT<edm::ValueMap<reco::GsfElectronRef> > valMapPFCandToEle_;

 std::string valueMapPFCandPhoton_;
 std::string valueMapPhoPFCandIso_;
 std::string valueMapPFCandEle_;
 std::string valueMapElePFCandIso_;

 PfBlockBasedIsolation* thePFBlockBasedIsolation_;

};

#endif

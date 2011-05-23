#ifndef RecoParticleFlow_PFProducer_EgammaPFLinker_h
#define RecoParticleFlow_PFProducer_EgammaPFLinker_h

/// Fills the GsfElectron Ref into the PFCandidate
/// Produces the ValueMap <GsfElectronRef,PFCandidateRef>
/// F. Beaudette 8 March 2011

#include <iostream>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include <string>

class EgammaPFLinker : public edm::EDProducer {
 public:

  explicit EgammaPFLinker(const edm::ParameterSet&);

  ~EgammaPFLinker();
  
  virtual void produce(edm::Event&, const edm::EventSetup&);

  virtual void beginRun(edm::Run& run,const edm::EventSetup & es);

 private:
  bool fetchCandidateCollection(edm::Handle<reco::PFCandidateCollection>& c, 
				const edm::InputTag& tag, 
				const edm::Event& iEvent) const ;
  
  bool fetchGsfElectronCollection(edm::Handle<reco::GsfElectronCollection>& c, 
				  const edm::InputTag& tag, 
				  const edm::Event& iEvent) const ;

  bool fetchPhotonCollection(edm::Handle<reco::PhotonCollection>& c, 
				  const edm::InputTag& tag, 
				  const edm::Event& iEvent) const ;
  
  void fillValueMap(edm::Handle<reco::GsfElectronCollection>& c,
		    const edm::OrphanHandle<reco::PFCandidateCollection> & pfOrphanHandle,
		    const edm::Handle<reco::PFCandidateCollection> & pfHandle,
		    edm::ValueMap<reco::PFCandidatePtr>::Filler & filler) const;

  void fillValueMap(edm::Handle<reco::PhotonCollection>& c,
		    const edm::OrphanHandle<reco::PFCandidateCollection> & pfOrphanHandle,
		    const edm::Handle<reco::PFCandidateCollection> & pfHandle,
		    edm::ValueMap<reco::PFCandidatePtr>::Filler & filler) const;

 private:
 
  /// Input PFCandidates
  edm::InputTag       inputTagPFCandidates_;

  /// Input GsfElectrons
  edm::InputTag       inputTagGsfElectrons_;

  /// Input Photons
  edm::InputTag       inputTagPhotons_;


  /// name of output collection of PFCandidate
  std::string nameOutputPF_;

  /// name of output ValueMap electrons
  std::string nameOutputElectronsPF_;

  /// name of output ValueMap photons
  std::string nameOutputPhotonsPF_;

  /// map GsfElectron PFCandidate (index)
  std::map<reco::GsfElectronRef,unsigned> electronCandidateMap_;

  /// map Photon PFCandidate (index)
  std::map<reco::PhotonRef,unsigned> photonCandidateMap_;

  /// Flags - if true: References will be towards new collection ; if false to the original one
  bool producePFCandidates_;
  
};

#endif

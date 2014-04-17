#ifndef RecoParticleFlow_PFProducer_PFLinker_h
#define RecoParticleFlow_PFProducer_PFLinker_h

/** \class PFLinker
 *  Producer meant for the Post PF reconstruction.
 *
 *  Fills the GsfElectron, Photon and Muon Ref into the PFCandidate
 *  Produces the ValueMap between GsfElectronRef/Photon/Mupns with PFCandidateRef
 *
 *  \author R. Bellan - UCSB <riccardo.bellan@cern.ch>, F. Beaudette - CERN <Florian.Beaudette@cern.ch>
 */

#include <iostream>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/MuonReco/interface/MuonToMuonMap.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

#include <string>

class PFLinker : public edm::stream::EDProducer<> {
 public:

  explicit PFLinker(const edm::ParameterSet&);

  ~PFLinker();
  
  virtual void produce(edm::Event&, const edm::EventSetup&) override;

 private:
  template<typename TYPE>
    edm::ValueMap<reco::PFCandidatePtr> fillValueMap(edm::Event & event,
						     std::string label,
						     edm::Handle<TYPE>& inputObjCollection,
						     const std::map<edm::Ref<TYPE>, reco::PFCandidatePtr> & mapToTheCandidate,
						     const edm::OrphanHandle<reco::PFCandidateCollection> & newPFCandColl) const;    
  
 private:
 
  /// Input PFCandidates
  std::vector<edm::EDGetTokenT<reco::PFCandidateCollection> >       inputTagPFCandidates_;

  /// Input GsfElectrons
  edm::EDGetTokenT<reco::GsfElectronCollection>    inputTagGsfElectrons_;

  /// Input Photons
  edm::EDGetTokenT<reco::PhotonCollection>         inputTagPhotons_;

  /// Input Muons
  edm::InputTag muonTag_;
  edm::EDGetTokenT<reco::MuonCollection>      inputTagMuons_;
  edm::EDGetTokenT<reco::MuonToMuonMap> inputTagMuonMap_;
  /// name of output collection of PFCandidate
  std::string nameOutputPF_;

  /// name of output ValueMap electrons
  std::string nameOutputElectronsPF_;

  /// name of output ValueMap photons
  std::string nameOutputPhotonsPF_;

  /// name of output merged ValueMap
  std::string nameOutputMergedPF_;

  /// Flags - if true: References will be towards new collection ; if false to the original one
  bool producePFCandidates_;

  /// Set muon refs and produce the value map?
  bool fillMuonRefs_;

};

DEFINE_FWK_MODULE(PFLinker);

#endif

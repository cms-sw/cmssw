#ifndef RecoHI_HiJetAlgos_HiFJRhoFlowModulationProducer_h
#define RecoHI_HiJetAlgos_HiFJRhoFlowModulationProducer_h

// user include files
#include "DataFormats/HeavyIonEvent/interface/EvtPlane.h"
#include "DataFormats/JetReco/interface/JetCollection.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

class HiFJRhoFlowModulationProducer : public edm::EDProducer {
public:
  explicit HiFJRhoFlowModulationProducer(const edm::ParameterSet&);
  ~HiFJRhoFlowModulationProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  bool doEvtPlane_;
  bool doFreePlaneFit_;
  bool doJettyExclusion_;
  int evtPlaneLevel_;
  edm::EDGetTokenT<reco::JetView> jetTag_;
  edm::EDGetTokenT<reco::PFCandidateCollection> pfCandsToken_;
  edm::EDGetTokenT<reco::EvtPlaneCollection> evtPlaneToken_;

  float* hiEvtPlane;
};

#endif

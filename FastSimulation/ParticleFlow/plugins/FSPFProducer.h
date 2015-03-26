#ifndef FastSimulation_ParticleFlow_FSPFProducer_h_
#define FastSimulation_ParticleFlow_FSPFProducer_h_

// system include files
#include <string>

// framework include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

class PFCandidate;

class FSPFProducer : public edm::stream::EDProducer <> {
 public:
  explicit FSPFProducer(const edm::ParameterSet&);
  ~FSPFProducer();
  
  virtual void produce(edm::Event&, const edm::EventSetup&) override;

 private:

  edm::InputTag labelPFCandidateCollection_;
  
  double par1, par2;
  double barrel_th, endcap_th, middle_th;

  bool pfPatchInHF;
  double HF_Ratio;
  std::vector<double> EM_HF_ScaleFactor;
  
  double energy_threshold(double eta);

  edm::EDGetTokenT<reco::PFCandidateCollection> pfCandidateToken;
};

#endif

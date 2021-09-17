#ifndef GeneratorInterface_RivetInterface_ParticleLevelProducer_H
#define GeneratorInterface_RivetInterface_ParticleLevelProducer_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"

#include "Rivet/AnalysisHandler.hh"
#include "GeneratorInterface/RivetInterface/interface/RivetAnalysis.h"

class ParticleLevelProducer : public edm::one::EDProducer<edm::one::SharedResources> {
public:
  ParticleLevelProducer(const edm::ParameterSet& pset);
  ~ParticleLevelProducer() override {}
  void produce(edm::Event& event, const edm::EventSetup& eventSetup) override;

private:
  void addGenJet(Rivet::Jet jet,
                 std::unique_ptr<reco::GenJetCollection>& jets,
                 std::unique_ptr<reco::GenParticleCollection>& consts,
                 edm::RefProd<reco::GenParticleCollection>& constsRefHandle,
                 int& iConstituent,
                 std::unique_ptr<reco::GenParticleCollection>& tags,
                 edm::RefProd<reco::GenParticleCollection>& tagsRefHandle,
                 int& iTag);

  template <typename T>
  reco::Candidate::LorentzVector p4(const T& p) const {
    return reco::Candidate::LorentzVector(p.px(), p.py(), p.pz(), p.energy());
  }

  const edm::EDGetTokenT<edm::HepMCProduct> srcToken_;
  const edm::ParameterSet pset_;

  reco::Particle::Point genVertex_;

  Rivet::RivetAnalysis* rivetAnalysis_ = nullptr;
  std::unique_ptr<Rivet::AnalysisHandler> analysisHandler_;
};

#endif

#ifndef RecoJets_JetProducers_plugins_PileupJetIDProducer_h
#define RecoJets_JetProducers_plugins_PileupJetIDProducer_h

// -*- C++ -*-
//
// Package:    PileupJetIdProducer
// Class:      PileupJetIdProducer
//
/**\class PileupJetIdProducer PileupJetIdProducer.cc CMGTools/PileupJetIdProducer/src/PileupJetIdProducer.cc

Description: Produces a value map of jet --> pileup jet ID

Implementation:
[Notes on implementation]
*/
//
// Original Author:  Pasquale Musella,40 2-A12,+41227671706,
//         Created:  Wed Apr 18 15:48:47 CEST 2012
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoJets/JetProducers/interface/PileupJetIdAlgo.h"

#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/JetReco/interface/PileupJetIdentifier.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"
#include "CondFormats/JetMETObjects/interface/FactorizedJetCorrector.h"
#include "JetMETCorrections/Objects/interface/JetCorrectionsRecord.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"

// ------------------------------------------------------------------------------------------

class GBRForestsAndConstants {
public:
  GBRForestsAndConstants(edm::ParameterSet const&);

  std::vector<PileupJetIdAlgo::AlgoGBRForestsAndConstants> const& vAlgoGBRForestsAndConstants() const {
    return vAlgoGBRForestsAndConstants_;
  }

  bool runMvas() const { return runMvas_; }
  bool produceJetIds() const { return produceJetIds_; }
  bool inputIsCorrected() const { return inputIsCorrected_; }
  bool applyJec() const { return applyJec_; }
  std::string const& jec() const { return jec_; }
  bool residualsFromTxt() const { return residualsFromTxt_; }
  edm::FileInPath const& residualsTxt() const { return residualsTxt_; }
  bool applyConstituentWeight() const { return applyConstituentWeight_; }

private:
  std::vector<PileupJetIdAlgo::AlgoGBRForestsAndConstants> vAlgoGBRForestsAndConstants_;

  bool runMvas_;
  bool produceJetIds_;
  bool inputIsCorrected_;
  bool applyJec_;
  std::string jec_;
  bool residualsFromTxt_;
  edm::FileInPath residualsTxt_;
  bool applyConstituentWeight_;
};

class PileupJetIdProducer : public edm::stream::EDProducer<edm::GlobalCache<GBRForestsAndConstants>> {
public:
  explicit PileupJetIdProducer(const edm::ParameterSet&, GBRForestsAndConstants const*);
  ~PileupJetIdProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  static std::unique_ptr<GBRForestsAndConstants> initializeGlobalCache(edm::ParameterSet const& pset) {
    return std::make_unique<GBRForestsAndConstants>(pset);
  }

  static void globalEndJob(GBRForestsAndConstants*) {}

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  void initJetEnergyCorrector(const edm::EventSetup& iSetup, bool isData);

  std::vector<std::pair<std::string, std::unique_ptr<PileupJetIdAlgo>>> algos_;

  std::unique_ptr<FactorizedJetCorrector> jecCor_;
  std::vector<JetCorrectorParameters> jetCorPars_;

  edm::ValueMap<float> constituentWeights_;
  edm::EDGetTokenT<edm::ValueMap<float>> input_constituent_weights_token_;
  edm::EDGetTokenT<edm::View<reco::Jet>> input_jet_token_;
  edm::EDGetTokenT<reco::VertexCollection> input_vertex_token_;
  edm::EDGetTokenT<edm::ValueMap<StoredPileupJetIdentifier>> input_vm_pujetid_token_;
  edm::EDGetTokenT<double> input_rho_token_;
  edm::ESGetToken<JetCorrectorParametersCollection, JetCorrectionsRecord> parameters_token_;
};

#endif

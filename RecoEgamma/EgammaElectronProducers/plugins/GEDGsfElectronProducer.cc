#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateEGammaExtra.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateEGammaExtraFwd.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/GsfElectronAlgo.h"

#include "GsfElectronBaseProducer.h"

#include <iostream>
#include <string>
#include <map>

class GEDGsfElectronProducer : public GsfElectronBaseProducer {
public:
  explicit GEDGsfElectronProducer(const edm::ParameterSet&, const GsfElectronAlgo::HeavyObjectCache*);
  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  edm::EDGetTokenT<reco::PFCandidateCollection> egmPFCandidateCollection_;
  std::map<reco::GsfTrackRef, reco::GsfElectron::MvaInput> gsfMVAInputMap_;
  std::map<reco::GsfTrackRef, reco::GsfElectron::MvaOutput> gsfMVAOutputMap_;

private:
  void fillGsfElectronValueMap(edm::Handle<std::vector<reco::PFCandidate>> const& pfCandidatesHandle,
                               edm::OrphanHandle<reco::GsfElectronCollection> const& orphanHandle,
                               edm::ValueMap<reco::GsfElectronRef>::Filler& filler);
  void matchWithPFCandidates(std::vector<reco::PFCandidate> const& pfCandidates);
  void setMVAOutputs(reco::GsfElectronCollection& electrons,
                     const GsfElectronAlgo::HeavyObjectCache*,
                     const std::map<reco::GsfTrackRef, reco::GsfElectron::MvaOutput>& mvaOutputs,
                     reco::VertexCollection const& vertices) const;
};

using namespace reco;

GEDGsfElectronProducer::GEDGsfElectronProducer(const edm::ParameterSet& cfg,
                                               const GsfElectronAlgo::HeavyObjectCache* hoc)
    : GsfElectronBaseProducer(cfg, hoc),
      egmPFCandidateCollection_(
          consumes<reco::PFCandidateCollection>(cfg.getParameter<edm::InputTag>("egmPFCandidatesTag"))) {
  produces<edm::ValueMap<reco::GsfElectronRef>>();
}

// ------------ method called to produce the data  ------------
void GEDGsfElectronProducer::produce(edm::Event& event, const edm::EventSetup& setup) {
  auto pfCandidatesHandle = event.getHandle(egmPFCandidateCollection_);
  matchWithPFCandidates(*pfCandidatesHandle);
  auto electrons = algo_->completeElectrons(event, setup, globalCache());
  setMVAOutputs(electrons, globalCache(), gsfMVAOutputMap_, event.get(inputCfg_.vtxCollectionTag));
  for (auto& el : electrons)
    el.setMvaInput(gsfMVAInputMap_.find(el.gsfTrack())->second);  // set MVA inputs
  auto orphanHandle = fillEvent(electrons, event);

  // ValueMap
  auto valMap_p = std::make_unique<edm::ValueMap<reco::GsfElectronRef>>();
  edm::ValueMap<reco::GsfElectronRef>::Filler valMapFiller(*valMap_p);
  fillGsfElectronValueMap(pfCandidatesHandle, orphanHandle, valMapFiller);
  valMapFiller.fill();
  event.put(std::move(valMap_p));
  // Done with the ValueMap
}

void GEDGsfElectronProducer::fillGsfElectronValueMap(
    edm::Handle<std::vector<reco::PFCandidate>> const& pfCandidatesHandle,
    edm::OrphanHandle<reco::GsfElectronCollection> const& orphanHandle,
    edm::ValueMap<reco::GsfElectronRef>::Filler& filler) {
  //Loop over the collection of PFFCandidates
  std::vector<reco::GsfElectronRef> values;

  for (auto const& pfCandidate : *pfCandidatesHandle) {
    reco::GsfElectronRef myRef;
    // First check that the GsfTrack is non null
    if (pfCandidate.gsfTrackRef().isNonnull()) {
      // now look for the corresponding GsfElectron
      const auto itcheck = std::find_if(orphanHandle->begin(), orphanHandle->end(), [&pfCandidate](const auto& ele) {
        return (ele.gsfTrack() == pfCandidate.gsfTrackRef());
      });
      if (itcheck != orphanHandle->end()) {
        // Build the Ref from the handle and the index
        myRef = reco::GsfElectronRef(orphanHandle, itcheck - orphanHandle->begin());
      }
    }
    values.push_back(myRef);
  }
  filler.insert(pfCandidatesHandle, values.begin(), values.end());
}

// Something more clever has to be found. The collections are small, so the timing is not
// an issue here; but it is clearly suboptimal

void GEDGsfElectronProducer::matchWithPFCandidates(std::vector<reco::PFCandidate> const& pfCandidates) {
  gsfMVAInputMap_.clear();
  gsfMVAOutputMap_.clear();

  //Loop over the collection of PFFCandidates
  for (auto const& pfCand : pfCandidates) {
    reco::GsfElectronRef myRef;
    // First check that the GsfTrack is non null
    if (pfCand.gsfTrackRef().isNonnull()) {
      reco::GsfElectron::MvaOutput myMvaOutput;
      // at the moment, undefined
      myMvaOutput.status = pfCand.egammaExtraRef()->electronStatus();
      gsfMVAOutputMap_[pfCand.gsfTrackRef()] = myMvaOutput;

      reco::GsfElectron::MvaInput myMvaInput;
      myMvaInput.earlyBrem = pfCand.egammaExtraRef()->mvaVariable(reco::PFCandidateEGammaExtra::MVA_FirstBrem);
      myMvaInput.lateBrem = pfCand.egammaExtraRef()->mvaVariable(reco::PFCandidateEGammaExtra::MVA_LateBrem);
      myMvaInput.deltaEta =
          pfCand.egammaExtraRef()->mvaVariable(reco::PFCandidateEGammaExtra::MVA_DeltaEtaTrackCluster);
      myMvaInput.sigmaEtaEta = pfCand.egammaExtraRef()->sigmaEtaEta();
      myMvaInput.hadEnergy = pfCand.egammaExtraRef()->hadEnergy();
      gsfMVAInputMap_[pfCand.gsfTrackRef()] = myMvaInput;
    }
  }
}

void GEDGsfElectronProducer::setMVAOutputs(reco::GsfElectronCollection& electrons,
                                           const GsfElectronAlgo::HeavyObjectCache* hoc,
                                           const std::map<reco::GsfTrackRef, reco::GsfElectron::MvaOutput>& mvaOutputs,
                                           reco::VertexCollection const& vertices) const {
  for (auto& el : electrons) {
    float mva_NIso_Value = hoc->sElectronMVAEstimator->mva(el, vertices);
    float mva_Iso_Value = hoc->iElectronMVAEstimator->mva(el, vertices.size());
    GsfElectron::MvaOutput mvaOutput;
    mvaOutput.mva_e_pi = mva_NIso_Value;
    mvaOutput.mva_Isolated = mva_Iso_Value;
    el.setMvaOutput(mvaOutput);
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(GEDGsfElectronProducer);

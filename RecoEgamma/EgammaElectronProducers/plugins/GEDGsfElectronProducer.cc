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
  void fillGsfElectronValueMap(edm::Event& event, edm::ValueMap<reco::GsfElectronRef>::Filler& filler);
  void matchWithPFCandidates(edm::Event& event);
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
  matchWithPFCandidates(event);
  reco::GsfElectronCollection electrons;
  algo_->completeElectrons(electrons, event, setup, globalCache());
  setMVAOutputs(electrons, globalCache(), gsfMVAOutputMap_, event.get(inputCfg_.vtxCollectionTag));
  for (auto& el : electrons)
    el.setMvaInput(gsfMVAInputMap_.find(el.gsfTrack())->second);  // set MVA inputs
  fillEvent(electrons, event);

  // ValueMap
  auto valMap_p = std::make_unique<edm::ValueMap<reco::GsfElectronRef>>();
  edm::ValueMap<reco::GsfElectronRef>::Filler valMapFiller(*valMap_p);
  fillGsfElectronValueMap(event, valMapFiller);
  valMapFiller.fill();
  event.put(std::move(valMap_p));
  // Done with the ValueMap
}

void GEDGsfElectronProducer::fillGsfElectronValueMap(edm::Event& event,
                                                     edm::ValueMap<reco::GsfElectronRef>::Filler& filler) {
  // Read the collection of PFCandidates
  edm::Handle<reco::PFCandidateCollection> pfCandidates;

  bool found = event.getByToken(egmPFCandidateCollection_, pfCandidates);
  if (!found) {
    edm::LogError("GEDGsfElectronProducer") << " cannot get PFCandidates! ";
  }

  //Loop over the collection of PFFCandidates
  reco::PFCandidateCollection::const_iterator it = pfCandidates->begin();
  reco::PFCandidateCollection::const_iterator itend = pfCandidates->end();
  std::vector<reco::GsfElectronRef> values;

  for (; it != itend; ++it) {
    reco::GsfElectronRef myRef;
    // First check that the GsfTrack is non null
    if (it->gsfTrackRef().isNonnull()) {
      // now look for the corresponding GsfElectron
      const auto itcheck = std::find_if(orphanHandle()->begin(), orphanHandle()->end(), [it](const auto& ele) {
        return (ele.gsfTrack() == it->gsfTrackRef());
      });
      if (itcheck != orphanHandle()->end()) {
        // Build the Ref from the handle and the index
        myRef = reco::GsfElectronRef(orphanHandle(), itcheck - orphanHandle()->begin());
      }
    }
    values.push_back(myRef);
  }
  filler.insert(pfCandidates, values.begin(), values.end());
}

// Something more clever has to be found. The collections are small, so the timing is not
// an issue here; but it is clearly suboptimal

void GEDGsfElectronProducer::matchWithPFCandidates(edm::Event& event) {
  gsfMVAInputMap_.clear();
  gsfMVAOutputMap_.clear();

  // Read the collection of PFCandidates
  edm::Handle<reco::PFCandidateCollection> pfCandidates;

  bool found = event.getByToken(egmPFCandidateCollection_, pfCandidates);
  if (!found) {
    edm::LogError("GEDGsfElectronProducer") << " cannot get PFCandidates! ";
  }

  //Loop over the collection of PFFCandidates
  reco::PFCandidateCollection::const_iterator it = pfCandidates->begin();
  reco::PFCandidateCollection::const_iterator itend = pfCandidates->end();

  for (; it != itend; ++it) {
    reco::GsfElectronRef myRef;
    // First check that the GsfTrack is non null
    if (it->gsfTrackRef().isNonnull()) {
      reco::GsfElectron::MvaOutput myMvaOutput;
      // at the moment, undefined
      myMvaOutput.status = it->egammaExtraRef()->electronStatus();
      gsfMVAOutputMap_[it->gsfTrackRef()] = myMvaOutput;

      reco::GsfElectron::MvaInput myMvaInput;
      myMvaInput.earlyBrem = it->egammaExtraRef()->mvaVariable(reco::PFCandidateEGammaExtra::MVA_FirstBrem);
      myMvaInput.lateBrem = it->egammaExtraRef()->mvaVariable(reco::PFCandidateEGammaExtra::MVA_LateBrem);
      myMvaInput.deltaEta = it->egammaExtraRef()->mvaVariable(reco::PFCandidateEGammaExtra::MVA_DeltaEtaTrackCluster);
      myMvaInput.sigmaEtaEta = it->egammaExtraRef()->sigmaEtaEta();
      myMvaInput.hadEnergy = it->egammaExtraRef()->hadEnergy();
      gsfMVAInputMap_[it->gsfTrackRef()] = myMvaInput;
    }
  }
}

void GEDGsfElectronProducer::setMVAOutputs(reco::GsfElectronCollection& electrons,
                                           const GsfElectronAlgo::HeavyObjectCache* hoc,
                                           const std::map<reco::GsfTrackRef, reco::GsfElectron::MvaOutput>& mvaOutputs,
                                           reco::VertexCollection const& vertices) const {
  for (auto el = electrons.begin(); el != electrons.end(); el++) {
    float mva_NIso_Value = hoc->sElectronMVAEstimator->mva(*el, vertices);
    float mva_Iso_Value = hoc->iElectronMVAEstimator->mva(*el, vertices.size());
    GsfElectron::MvaOutput mvaOutput;
    mvaOutput.mva_e_pi = mva_NIso_Value;
    mvaOutput.mva_Isolated = mva_Iso_Value;
    el->setMvaOutput(mvaOutput);
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(GEDGsfElectronProducer);

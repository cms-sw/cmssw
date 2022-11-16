//
//
//
//

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/PatCandidates/interface/Muon.h"
#include "PhysicsTools/PatAlgos/interface/BaseMVAValueMapProducer.h"
#include <vector>

class EvaluateMuonMVAID : public BaseMVAValueMapProducer<pat::Muon> {
public:
  explicit EvaluateMuonMVAID(const edm::ParameterSet& iConfig, const BaseMVACache* cache)
      : BaseMVAValueMapProducer<pat::Muon>(iConfig, cache) {}

  void fillAdditionalVariables(const pat::Muon& m) override {
    float norm_chi2;
    float n_Valid_hits;
    if (m.globalTrack().isNonnull()) {
      norm_chi2 = m.globalTrack()->normalizedChi2();
      n_Valid_hits = m.globalTrack()->hitPattern().numberOfValidMuonHits();
    } else if (m.innerTrack().isNonnull()) {
      norm_chi2 = m.innerTrack()->normalizedChi2();
      n_Valid_hits = m.innerTrack()->hitPattern().numberOfValidMuonHits();
    } else {
      norm_chi2 = -99;
      n_Valid_hits = -99;
    }

    this->setValue("Muon_norm_chi2_extended", norm_chi2);
    this->setValue("Muon_n_Valid_hits_extended", n_Valid_hits);
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc = BaseMVAValueMapProducer<pat::Muon>::getDescription();
    descriptions.add("EvaluateMuonMVAID", desc);
  }
};

//define this as a plug-in
DEFINE_FWK_MODULE(EvaluateMuonMVAID);

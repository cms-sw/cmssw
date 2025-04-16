/// original authors: RK18 team
#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "KinVtxFitter.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "helper.h"

template <typename Lepton>
class DiLeptonBuilder : public edm::global::EDProducer<> {
  // perhaps we need better structure here (begin run etc)
public:
  typedef std::vector<Lepton> LeptonCollection;
  typedef std::vector<reco::TransientTrack> TransientTrackCollection;

  explicit DiLeptonBuilder(const edm::ParameterSet &cfg)
      : l1_selection_{cfg.getParameter<std::string>("lep1Selection")},
        l2_selection_{cfg.getParameter<std::string>("lep2Selection")},
        pre_vtx_selection_{cfg.getParameter<std::string>("preVtxSelection")},
        post_vtx_selection_{cfg.getParameter<std::string>("postVtxSelection")},
        src_{consumes<LeptonCollection>(cfg.getParameter<edm::InputTag>("src"))},
        ttracks_src_{consumes<TransientTrackCollection>(cfg.getParameter<edm::InputTag>("transientTracksSrc"))} {
    produces<pat::CompositeCandidateCollection>("SelectedDiLeptons");
  }

  ~DiLeptonBuilder() override {}

  void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;

private:
  const StringCutObjectSelector<Lepton> l1_selection_;                         // cut on leading lepton
  const StringCutObjectSelector<Lepton> l2_selection_;                         // cut on sub-leading lepton
  const StringCutObjectSelector<pat::CompositeCandidate> pre_vtx_selection_;   // cut on the di-lepton before the SV fit
  const StringCutObjectSelector<pat::CompositeCandidate> post_vtx_selection_;  // cut on the di-lepton after the SV fit
  const edm::EDGetTokenT<LeptonCollection> src_;
  const edm::EDGetTokenT<TransientTrackCollection> ttracks_src_;
};

template <typename Lepton>
void DiLeptonBuilder<Lepton>::produce(edm::StreamID, edm::Event &evt, edm::EventSetup const &) const {
  // input
  edm::Handle<LeptonCollection> leptons;
  evt.getByToken(src_, leptons);

  edm::Handle<TransientTrackCollection> ttracks;
  evt.getByToken(ttracks_src_, ttracks);

  // output
  std::unique_ptr<pat::CompositeCandidateCollection> ret_value(new pat::CompositeCandidateCollection());

  for (size_t l1_idx = 0; l1_idx < leptons->size(); ++l1_idx) {
    edm::Ptr<Lepton> l1_ptr(leptons, l1_idx);
    if (!l1_selection_(*l1_ptr))
      continue;

    for (size_t l2_idx = l1_idx + 1; l2_idx < leptons->size(); ++l2_idx) {
      edm::Ptr<Lepton> l2_ptr(leptons, l2_idx);
      if (!l2_selection_(*l2_ptr))
        continue;

      pat::CompositeCandidate lepton_pair;
      lepton_pair.setP4(l1_ptr->p4() + l2_ptr->p4());
      lepton_pair.setCharge(l1_ptr->charge() + l2_ptr->charge());
      lepton_pair.addUserFloat("lep_deltaR", reco::deltaR(*l1_ptr, *l2_ptr));

      // Put the lepton passing the corresponding selection
      lepton_pair.addUserInt("l1_idx", l1_idx);
      lepton_pair.addUserInt("l2_idx", l2_idx);

      // Use UserCands as they should not use memory but keep the Ptr itself
      lepton_pair.addUserCand("l1", l1_ptr);
      lepton_pair.addUserCand("l2", l2_ptr);
      if (!pre_vtx_selection_(lepton_pair))
        continue;  // before making the SV, cut on the info we have

      KinVtxFitter fitter(
          {ttracks->at(l1_idx), ttracks->at(l2_idx)}, {l1_ptr->mass(), l2_ptr->mass()}, {bph::LEP_SIGMA, bph::LEP_SIGMA}
          // some small sigma for the particle mass
      );
      if (!fitter.success())
        continue;
      lepton_pair.setVertex(
          reco::Candidate::Point(fitter.fitted_vtx().x(), fitter.fitted_vtx().y(), fitter.fitted_vtx().z()));
      lepton_pair.addUserInt("sv_ok", fitter.success() ? 1 : 0);
      lepton_pair.addUserFloat("sv_chi2", fitter.chi2());
      lepton_pair.addUserFloat("sv_ndof", fitter.dof());  // float??
      lepton_pair.addUserFloat("sv_prob", fitter.prob());
      lepton_pair.addUserFloat("fitted_mass", fitter.success() ? fitter.fitted_candidate().mass() : -1);
      lepton_pair.addUserFloat(
          "fitted_massErr",
          fitter.success() ? sqrt(fitter.fitted_candidate().kinematicParametersError().matrix()(6, 6)) : -1);
      lepton_pair.addUserFloat("vtx_x", lepton_pair.vx());
      lepton_pair.addUserFloat("vtx_y", lepton_pair.vy());
      lepton_pair.addUserFloat("vtx_z", lepton_pair.vz());

      // if needed, add here more stuff

      // cut on the SV info
      if (!post_vtx_selection_(lepton_pair))
        continue;
      ret_value->push_back(lepton_pair);
    }
  }

  evt.put(std::move(ret_value), "SelectedDiLeptons");
}

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
typedef DiLeptonBuilder<pat::Muon> DiMuonBuilder;
typedef DiLeptonBuilder<pat::Electron> DiElectronBuilder;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DiMuonBuilder);
DEFINE_FWK_MODULE(DiElectronBuilder);

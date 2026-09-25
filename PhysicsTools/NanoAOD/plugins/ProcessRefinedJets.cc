#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "CommonTools/Utils/interface/StringObjectFunction.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <cmath>
#include <memory>
#include <string>
#include <vector>

//This producer embeds the final refined FastSim jet pT values
//so that they can be used as input to the pT-based sorting routine, as well
//as to implement the Type-1 MET correction based on the refined jets.
class ProcessRefinedJets : public edm::global::EDProducer<> {
public:
  explicit ProcessRefinedJets(const edm::ParameterSet &iConfig)
      : jetsToken_(consumes<edm::View<pat::Jet>>(iConfig.getParameter<edm::InputTag>("jets"))),
        refinedPtName_(iConfig.getParameter<std::string>("refinedPtName")),
        maskBtagName_(iConfig.getParameter<std::string>("maskBtagName")),
        ptFinalName_(iConfig.getParameter<std::string>("ptFinalName")),
        ptUnrefinedName_(iConfig.getParameter<std::string>("ptUnrefinedName")),
        minGenJetPt_(iConfig.existsAs<double>("minGenJetPt") ? iConfig.getParameter<double>("minGenJetPt") : -1.) {
    if (iConfig.existsAs<std::vector<std::string>>("taggerNames")) {
      taggerNames_ = iConfig.getParameter<std::vector<std::string>>("taggerNames");
      const auto expressions = iConfig.getParameter<std::vector<std::string>>("rawTaggerExpressions");
      if (expressions.size() != taggerNames_.size())
        throw cms::Exception("Configuration") << "Each refined tagger requires its raw expression";
      for (const auto &expression : expressions)
        rawTaggers_.emplace_back(expression);
    }
    // Type-1 MET correction is optional
    if (iConfig.existsAs<edm::InputTag>("met")) {
      metToken_ = consumes<edm::View<pat::MET>>(iConfig.getParameter<edm::InputTag>("met"));
      doMET_ = true;
    } else {
      doMET_ = false;
    }

    produces<std::vector<pat::Jet>>();

    if (doMET_) {
      // refined MET made available in python
      produces<std::vector<pat::MET>>("Refined");
    }
  }

  ~ProcessRefinedJets() override = default;

  void produce(edm::StreamID, edm::Event &iEvent, const edm::EventSetup &) const override {
    edm::Handle<edm::View<pat::Jet>> hJets;
    iEvent.getByToken(jetsToken_, hJets);

    auto outJets = std::make_unique<std::vector<pat::Jet>>();
    outJets->reserve(hJets->size());

    // Sum over (pt_orig - pt_final) for MET correction
    double sumDeltaPx = 0.;
    double sumDeltaPy = 0.;

    for (auto const &jIn : *hJets) {
      pat::Jet j(jIn);

      const double pt_orig = j.pt();
      const double phi = j.phi();

      // mask: BvsAll > 0
      const float bvsAll = j.bDiscriminator(maskBtagName_);
      bool refine = (bvsAll > 0.f);
      if (minGenJetPt_ >= 0.) {
        const auto &gen = j.genJetFwdRef().backRef();
        refine =
            refine && gen.isNonnull() && gen.isAvailable() && std::isfinite(gen->pt()) && gen->pt() >= minGenJetPt_;
        // Share the decision with the Nano taggers, pT sorting, and MET.
        j.addUserInt("fastSimRefinementApplied", refine);
      }

      const double pt_ref = j.hasUserFloat(refinedPtName_) ? static_cast<double>(j.userFloat(refinedPtName_)) : pt_orig;

      const double pt_final = refine ? pt_ref : pt_orig;
      if (minGenJetPt_ >= 0. && (!std::isfinite(pt_final) || pt_final <= 0.))
        throw cms::Exception("InvalidRefinedPt") << "Nonphysical final jet pT: " << pt_final;

      // Select scores here; raw Nano expressions can themselves contain conditionals.
      for (unsigned int i = 0; i < taggerNames_.size(); ++i) {
        const float value = refine ? j.userFloat(taggerNames_[i] + "refined") : rawTaggers_[i](jIn);
        if (!std::isfinite(value) || (refine && (value < 0.f || value > 1.f)))
          throw cms::Exception("InvalidRefinedScore") << taggerNames_[i] << ": " << value;
        j.addUserFloat("fastSimFinal_" + taggerNames_[i], value);
      }

      // store unrefined pt if requested
      if (!ptUnrefinedName_.empty()) {
        j.addUserFloat(ptUnrefinedName_, static_cast<float>(pt_orig));
      }

      // store final pt used for Nano, needed for the sorting
      j.addUserFloat(ptFinalName_, static_cast<float>(pt_final));

      // MET delta: add unrefined jets, subtract final jets
      // This is equivalent to adding (pt_orig - pt_final) in the jet direction.
      const double dpt = pt_orig - pt_final;
      sumDeltaPx += dpt * std::cos(phi);
      sumDeltaPy += dpt * std::sin(phi);

      outJets->push_back(std::move(j));
    }

    iEvent.put(std::move(outJets));

    // Optionally correct MET
    if (doMET_) {
      edm::Handle<edm::View<pat::MET>> hMET;
      iEvent.getByToken(metToken_, hMET);

      auto outMET = std::make_unique<std::vector<pat::MET>>();
      outMET->reserve(hMET->size());

      for (auto const &mIn : *hMET) {
        pat::MET m(mIn);

        const double pxOrig = m.px();
        const double pyOrig = m.py();
        const double ptOrig = m.pt();
        const double phiOrig = m.phi();

        const double pxFinal = pxOrig + sumDeltaPx;
        const double pyFinal = pyOrig + sumDeltaPy;
        const double ptFinal = std::sqrt(pxFinal * pxFinal + pyFinal * pyFinal);
        const double phiFinal = std::atan2(pyFinal, pxFinal);

        m.addUserFloat("pt_unrefined", static_cast<float>(ptOrig));
        m.addUserFloat("phi_unrefined", static_cast<float>(phiOrig));

        // stash final values, needed for sorting.
        m.addUserFloat("pt_final", static_cast<float>(ptFinal));
        m.addUserFloat("phi_final", static_cast<float>(phiFinal));

        // push back into pat::MET with reco::Candidate::LorentzVector ---
        reco::Candidate::LorentzVector p4(pxFinal, pyFinal, 0.0, ptFinal);
        m.setP4(p4);

        outMET->push_back(std::move(m));
      }
      iEvent.put(std::move(outMET), "Refined");
    }
  }

private:
  edm::EDGetTokenT<edm::View<pat::Jet>> jetsToken_;
  edm::EDGetTokenT<edm::View<pat::MET>> metToken_;
  bool doMET_ = false;

  std::string refinedPtName_;
  std::string maskBtagName_;
  std::string ptFinalName_;
  std::string ptUnrefinedName_;
  double minGenJetPt_;
  std::vector<std::string> taggerNames_;
  std::vector<StringObjectFunction<pat::Jet>> rawTaggers_;
};

DEFINE_FWK_MODULE(ProcessRefinedJets);

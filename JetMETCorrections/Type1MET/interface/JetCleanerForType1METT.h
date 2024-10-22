#ifndef JetMETCorrections_Type1MET_JetCleanerForType1METT_h
#define JetMETCorrections_Type1MET_JetCleanerForType1METT_h

/** \class JetCleanerForType1METT
 *
 * Clean jets for MET corrections and uncertainties (pt/eta/EM fraction and muons)
 * apply also JECs
 *
 * NOTE: class is templated to that it works with reco::PFJets as well as with pat::Jets of PF-type as input
 *
 * \authors Matthieu Marionneau ETH
 *
 *
 *
 */

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/METReco/interface/CorrMETData.h"
#include "DataFormats/PatCandidates/interface/Jet.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DataFormats/MuonReco/interface/Muon.h"

#include "JetMETCorrections/Type1MET/interface/JetCorrExtractorT.h"

#include <memory>

#include <string>
#include <type_traits>

namespace JetCleanerForType1MET_namespace {
  template <typename T, typename Textractor>
  class InputTypeCheckerT {
  public:
    void operator()(const T&) const {}  // no type-checking needed for reco::PFJet input
    bool isPatJet(const T&) const { return std::is_base_of<class pat::Jet, T>::value; }
  };

  template <typename T>
  class RawJetExtractorT  // this template is neccessary to support pat::Jets
  // (because pat::Jet->p4() returns the JES corrected, not the raw, jet momentum)
  // But it does not handle the muon removal!!!!! MM
  {
  public:
    RawJetExtractorT() {}
    reco::Candidate::LorentzVector operator()(const T& jet) const { return jet.p4(); }
  };
}  // namespace JetCleanerForType1MET_namespace

template <typename T, typename Textractor>
class JetCleanerForType1METT : public edm::stream::EDProducer<> {
public:
  explicit JetCleanerForType1METT(const edm::ParameterSet& cfg)
      : moduleLabel_(cfg.getParameter<std::string>("@module_label")),
        offsetCorrLabel_(""),
        skipMuonSelection_(nullptr) {
    token_ = consumes<std::vector<T>>(cfg.getParameter<edm::InputTag>("src"));

    offsetCorrLabel_ = cfg.getParameter<edm::InputTag>("offsetCorrLabel");
    offsetCorrToken_ = consumes<reco::JetCorrector>(offsetCorrLabel_);

    jetCorrLabel_ = cfg.getParameter<edm::InputTag>("jetCorrLabel");        //for MC
    jetCorrLabelRes_ = cfg.getParameter<edm::InputTag>("jetCorrLabelRes");  //for data
    jetCorrToken_ = mayConsume<reco::JetCorrector>(jetCorrLabel_);
    jetCorrResToken_ = mayConsume<reco::JetCorrector>(jetCorrLabelRes_);

    jetCorrEtaMax_ = cfg.getParameter<double>("jetCorrEtaMax");

    type1JetPtThreshold_ = cfg.getParameter<double>("type1JetPtThreshold");

    skipEM_ = cfg.getParameter<bool>("skipEM");
    if (skipEM_) {
      skipEMfractionThreshold_ = cfg.getParameter<double>("skipEMfractionThreshold");
    }

    skipMuons_ = cfg.getParameter<bool>("skipMuons");
    if (skipMuons_) {
      std::string skipMuonSelection_string = cfg.getParameter<std::string>("skipMuonSelection");
      skipMuonSelection_ = std::make_unique<StringCutObjectSelector<reco::Candidate>>(skipMuonSelection_string, true);
    }

    calcMuonSubtrRawPtAsValueMap_ = cfg.getParameter<bool>("calcMuonSubtrRawPtAsValueMap");

    produces<std::vector<T>>();
    if (calcMuonSubtrRawPtAsValueMap_)
      produces<edm::ValueMap<float>>("MuonSubtrRawPt");
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::string>("@module_label");
    desc.add<edm::InputTag>("src");
    desc.add<edm::InputTag>("offsetCorrLabel");
    desc.add<edm::InputTag>("jetCorrLabel");
    desc.add<edm::InputTag>("jetCorrLabelRes");
    desc.add<double>("jetCorrEtaMax", 9.9);
    desc.add<double>("type1JetPtThreshold");
    desc.add<bool>("skipEM");
    desc.add<double>("skipEMfractionThreshold");
    desc.add<bool>("skipMuons");
    desc.add<std::string>("skipMuonSelection");
    desc.add<bool>("calcMuonSubtrRawPtAsValueMap", false)
        ->setComment(
            "calculate muon-subtracted raw pt as a ValueMap for the input collection (only for selected jets, zero for "
            "others)");
    descriptions.addDefault(desc);
  }

private:
  void produce(edm::Event& evt, const edm::EventSetup& es) override {
    edm::Handle<reco::JetCorrector> jetCorr;
    //automatic switch for residual corrections
    if (evt.isRealData()) {
      jetCorrLabel_ = jetCorrLabelRes_;
      evt.getByToken(jetCorrResToken_, jetCorr);
    } else {
      evt.getByToken(jetCorrToken_, jetCorr);
    }

    typedef std::vector<T> JetCollection;
    edm::Handle<JetCollection> jets;
    evt.getByToken(token_, jets);

    //new collection
    std::unique_ptr<std::vector<T>> cleanedJets(new std::vector<T>());

    int numJets = jets->size();
    std::vector<float> muonSubtrRawPt(numJets, 0);

    for (int jetIndex = 0; jetIndex < numJets; ++jetIndex) {
      const T& jet = jets->at(jetIndex);

      const static JetCleanerForType1MET_namespace::InputTypeCheckerT<T, Textractor> checkInputType{};
      checkInputType(jet);

      double emEnergyFraction = jet.chargedEmEnergyFraction() + jet.neutralEmEnergyFraction();
      if (skipEM_ && emEnergyFraction > skipEMfractionThreshold_)
        continue;

      const static JetCleanerForType1MET_namespace::RawJetExtractorT<T> rawJetExtractor{};
      reco::Candidate::LorentzVector rawJetP4 = rawJetExtractor(jet);

      if (skipMuons_) {
        const std::vector<reco::CandidatePtr>& cands = jet.daughterPtrVector();
        for (std::vector<reco::CandidatePtr>::const_iterator cand = cands.begin(); cand != cands.end(); ++cand) {
          const reco::PFCandidate* pfcand = dynamic_cast<const reco::PFCandidate*>(cand->get());
          const reco::Candidate* mu =
              (pfcand != nullptr ? (pfcand->muonRef().isNonnull() ? pfcand->muonRef().get() : nullptr) : cand->get());
          if (mu != nullptr && (*skipMuonSelection_)(*mu)) {
            reco::Candidate::LorentzVector muonP4 = (*cand)->p4();
            rawJetP4 -= muonP4;
          }
        }
      }

      reco::Candidate::LorentzVector corrJetP4;
      if (checkInputType.isPatJet(jet)) {
        corrJetP4 = jetCorrExtractor_(jet, jetCorrLabel_.label(), jetCorrEtaMax_, &rawJetP4);
      } else {
        corrJetP4 = jetCorrExtractor_(jet, jetCorr.product(), jetCorrEtaMax_, &rawJetP4);
        if (corrJetP4.pt() < type1JetPtThreshold_)
          continue;
      }

      if (corrJetP4.pt() < type1JetPtThreshold_)
        continue;

      cleanedJets->push_back(jet);
      if (calcMuonSubtrRawPtAsValueMap_)
        muonSubtrRawPt[jetIndex] = rawJetP4.Pt();
    }

    evt.put(std::move(cleanedJets));

    if (calcMuonSubtrRawPtAsValueMap_) {
      std::unique_ptr<edm::ValueMap<float>> muonSubtrRawPtV(new edm::ValueMap<float>());
      edm::ValueMap<float>::Filler fillerMuonSubtrRawPt(*muonSubtrRawPtV);
      fillerMuonSubtrRawPt.insert(jets, muonSubtrRawPt.begin(), muonSubtrRawPt.end());
      fillerMuonSubtrRawPt.fill();
      evt.put(std::move(muonSubtrRawPtV), "MuonSubtrRawPt");
    }
  }

  std::string moduleLabel_;

  edm::EDGetTokenT<std::vector<T>> token_;

  edm::InputTag offsetCorrLabel_;
  edm::EDGetTokenT<reco::JetCorrector> offsetCorrToken_;  // e.g. 'ak5CaloJetL1Fastjet'
  edm::InputTag jetCorrLabel_;
  edm::InputTag jetCorrLabelRes_;
  edm::EDGetTokenT<reco::JetCorrector>
      jetCorrToken_;  // e.g. 'ak5CaloJetL1FastL2L3' (MC) / 'ak5CaloJetL1FastL2L3Residual' (Data)
  edm::EDGetTokenT<reco::JetCorrector>
      jetCorrResToken_;  // e.g. 'ak5CaloJetL1FastL2L3' (MC) / 'ak5CaloJetL1FastL2L3Residual' (Data)
  Textractor jetCorrExtractor_;

  double jetCorrEtaMax_;  // do not use JEC factors for |eta| above this threshold

  double type1JetPtThreshold_;  // threshold to distinguish between jets entering Type 1 MET correction
  // and jets entering "unclustered energy" sum
  // NOTE: threshold is applied on **corrected** jet energy (recommended default = 10 GeV)

  bool skipEM_;  // flag to exclude jets with large fraction of electromagnetic energy (electrons/photons)
  // from Type 1 + 2 MET corrections
  double skipEMfractionThreshold_;

  bool skipMuons_;  // flag to subtract momentum of muons (provided muons pass selection cuts) which are within jets
  // from jet energy before compute JECs/propagating JECs to Type 1 + 2 MET corrections
  std::unique_ptr<StringCutObjectSelector<reco::Candidate>> skipMuonSelection_;

  bool calcMuonSubtrRawPtAsValueMap_;  // calculate muon-subtracted raw pt as a ValueMap for the input collection (only for selected jets, zero for others)
};

#endif

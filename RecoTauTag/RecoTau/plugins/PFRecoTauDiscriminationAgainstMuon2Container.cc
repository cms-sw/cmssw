
/** \class PFRecoTauDiscriminationAgainstMuon2Container
 *
 * Compute tau Id. discriminator against muons.
 * 
 * \author Christian Veelken, LLR
 *
 */

#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"

#include "FWCore/Utilities/interface/Exception.h"

#include <FWCore/ParameterSet/interface/ConfigurationDescriptions.h>
#include <FWCore/ParameterSet/interface/ParameterSetDescription.h>

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonChamberMatch.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "RecoTauTag/RecoTau/interface/RecoTauMuonTools.h"
#include "RecoTauTag/RecoTau/plugins/PFRecoTauDiscriminationAgainstMuon2Helper.h"

#include <vector>
#include <string>
#include <iostream>
#include <atomic>

namespace {

  class PFRecoTauDiscriminationAgainstMuon2Container final : public PFTauDiscriminationContainerProducerBase {
  public:
    explicit PFRecoTauDiscriminationAgainstMuon2Container(const edm::ParameterSet& cfg)
        : PFTauDiscriminationContainerProducerBase(cfg), moduleLabel_(cfg.getParameter<std::string>("@module_label")) {
      auto const wpDefs = cfg.getParameter<std::vector<edm::ParameterSet>>("IDWPdefinitions");
      // check content of discriminatorOption and add as enum to avoid string comparison per event
      for (auto& wpDefsEntry : wpDefs) {
        std::string discriminatorOption_string = wpDefsEntry.getParameter<std::string>("discriminatorOption");
        int discOption;
        if (discriminatorOption_string == "loose")
          discOption = PFRecoTauDiscriminationAgainstMuonConfigSet::kLoose;
        else if (discriminatorOption_string == "medium")
          discOption = PFRecoTauDiscriminationAgainstMuonConfigSet::kMedium;
        else if (discriminatorOption_string == "tight")
          discOption = PFRecoTauDiscriminationAgainstMuonConfigSet::kTight;
        else if (discriminatorOption_string == "custom")
          discOption = PFRecoTauDiscriminationAgainstMuonConfigSet::kCustom;
        else
          throw edm::Exception(edm::errors::UnimplementedFeature)
              << " Invalid Configuration parameter 'discriminatorOption' = " << discriminatorOption_string << " !!\n";
        wpDefs_.push_back(
            PFRecoTauDiscriminationAgainstMuonConfigSet(discOption,
                                                        wpDefsEntry.getParameter<double>("HoPMin"),
                                                        wpDefsEntry.getParameter<int>("maxNumberOfMatches"),
                                                        wpDefsEntry.getParameter<bool>("doCaloMuonVeto"),
                                                        wpDefsEntry.getParameter<int>("maxNumberOfHitsLast2Stations")));
      }
      srcMuons_ = cfg.getParameter<edm::InputTag>("srcMuons");
      muons_token = consumes<reco::MuonCollection>(srcMuons_);
      dRmuonMatch_ = cfg.getParameter<double>("dRmuonMatch");
      dRmuonMatchLimitedToJetArea_ = cfg.getParameter<bool>("dRmuonMatchLimitedToJetArea");
      minPtMatchedMuon_ = cfg.getParameter<double>("minPtMatchedMuon");
      typedef std::vector<int> vint;
      maskMatchesDT_ = cfg.getParameter<vint>("maskMatchesDT");
      maskMatchesCSC_ = cfg.getParameter<vint>("maskMatchesCSC");
      maskMatchesRPC_ = cfg.getParameter<vint>("maskMatchesRPC");
      maskHitsDT_ = cfg.getParameter<vint>("maskHitsDT");
      maskHitsCSC_ = cfg.getParameter<vint>("maskHitsCSC");
      maskHitsRPC_ = cfg.getParameter<vint>("maskHitsRPC");
      verbosity_ = cfg.getParameter<int>("verbosity");
    }
    ~PFRecoTauDiscriminationAgainstMuon2Container() override {}

    void beginEvent(const edm::Event&, const edm::EventSetup&) override;

    reco::SingleTauDiscriminatorContainer discriminate(const reco::PFTauRef&) const override;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    std::string moduleLabel_;
    std::vector<PFRecoTauDiscriminationAgainstMuonConfigSet> wpDefs_;
    edm::InputTag srcMuons_;
    edm::Handle<reco::MuonCollection> muons_;
    edm::EDGetTokenT<reco::MuonCollection> muons_token;
    double dRmuonMatch_;
    bool dRmuonMatchLimitedToJetArea_;
    double minPtMatchedMuon_;
    std::vector<int> maskMatchesDT_;
    std::vector<int> maskMatchesCSC_;
    std::vector<int> maskMatchesRPC_;
    std::vector<int> maskHitsDT_;
    std::vector<int> maskHitsCSC_;
    std::vector<int> maskHitsRPC_;
    static std::atomic<unsigned int> numWarnings_;
    static constexpr unsigned int maxWarnings_ = 3;
    int verbosity_;
  };

  std::atomic<unsigned int> PFRecoTauDiscriminationAgainstMuon2Container::numWarnings_{0};

  void PFRecoTauDiscriminationAgainstMuon2Container::beginEvent(const edm::Event& evt, const edm::EventSetup& es) {
    if (!srcMuons_.label().empty()) {
      evt.getByToken(muons_token, muons_);
    }
  }

  reco::SingleTauDiscriminatorContainer PFRecoTauDiscriminationAgainstMuon2Container::discriminate(
      const reco::PFTauRef& pfTau) const {
    const reco::PFCandidatePtr& pfCand = pfTau->leadPFChargedHadrCand();
    auto helper = PFRecoTauDiscriminationAgainstMuon2Helper(verbosity_,
                                                            moduleLabel_,
                                                            srcMuons_.label().empty(),
                                                            minPtMatchedMuon_,
                                                            dRmuonMatch_,
                                                            dRmuonMatchLimitedToJetArea_,
                                                            numWarnings_,
                                                            maxWarnings_,
                                                            maskMatchesDT_,
                                                            maskMatchesCSC_,
                                                            maskMatchesRPC_,
                                                            maskHitsDT_,
                                                            maskHitsCSC_,
                                                            maskHitsRPC_,
                                                            muons_,
                                                            pfTau,
                                                            pfCand);

    reco::SingleTauDiscriminatorContainer result;
    for (auto const& wpDefsEntry : wpDefs_) {
      result.workingPoints.push_back(helper.eval(wpDefsEntry, pfTau));
      if (verbosity_)
        edm::LogPrint("PFTauAgainstMuon2") << "--> returning discriminatorValue = " << result.workingPoints.back();
    }
    return result;
  }

}  // namespace

void PFRecoTauDiscriminationAgainstMuon2Container::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // pfRecoTauDiscriminationAgainstMuon2Container
  edm::ParameterSetDescription desc;
  desc.add<std::vector<int>>("maskHitsRPC",
                             {
                                 0,
                                 0,
                                 0,
                                 0,
                             });
  desc.add<std::vector<int>>("maskMatchesRPC",
                             {
                                 0,
                                 0,
                                 0,
                                 0,
                             });
  desc.add<std::vector<int>>("maskMatchesCSC",
                             {
                                 1,
                                 0,
                                 0,
                                 0,
                             })
      ->setComment(
          "flags to mask/unmask DT, CSC and RPC chambers in individual muon stations. Segments and hits that are "
          "present in that muon station are ignored in case the 'mask' is set to 1. Per default only the innermost CSC "
          "chamber is ignored, as it is affected by spurious hits in high pile-up events.");
  desc.add<std::vector<int>>("maskHitsCSC",
                             {
                                 0,
                                 0,
                                 0,
                                 0,
                             });
  desc.add<edm::InputTag>("PFTauProducer", edm::InputTag("pfRecoTauProducer"));
  desc.add<int>("verbosity", 0);
  desc.add<std::vector<int>>("maskMatchesDT",
                             {
                                 0,
                                 0,
                                 0,
                                 0,
                             });
  desc.add<double>("minPtMatchedMuon", 5.0);
  desc.add<bool>("dRmuonMatchLimitedToJetArea", false);
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("BooleanOperator", "and");
    {
      edm::ParameterSetDescription psd1;
      psd1.add<double>("cut", 0.5);
      psd1.add<edm::InputTag>("Producer", edm::InputTag("pfRecoTauDiscriminationByLeadingTrackFinding"));
      psd0.addOptional<edm::ParameterSetDescription>("leadTrack", psd1);  // optional with default?
    }
    // Prediscriminants can be
    // Prediscriminants = noPrediscriminants,
    // as in RecoTauTag/Configuration/python/HPSPFTaus_cff.py
    //
    // and the definition is:
    // RecoTauTag/RecoTau/python/TauDiscriminatorTools.py
    // # Require no prediscriminants
    // noPrediscriminants = cms.PSet(
    //       BooleanOperator = cms.string("and"),
    //       )
    // -- so this is the minimum required definition
    // otherwise it inserts the leadTrack with Producer = InpuTag(...) and does not find the corresponding output during the run
    desc.add<edm::ParameterSetDescription>("Prediscriminants", psd0);
  }
  desc.add<std::vector<int>>("maskHitsDT",
                             {
                                 0,
                                 0,
                                 0,
                                 0,
                             });
  desc.add<double>("dRmuonMatch", 0.3);
  desc.add<edm::InputTag>("srcMuons", edm::InputTag("muons"))
      ->setComment("optional collection of muons to check for overlap with taus");

  edm::ParameterSetDescription desc_wp;
  desc_wp.add<std::string>("IDname");
  desc_wp.add<std::string>("discriminatorOption")
      ->setComment("available options are: 'loose', 'medium', 'tight' and 'custom'");
  desc_wp.add<double>("HoPMin");
  desc_wp.add<int>("maxNumberOfMatches")
      ->setComment("negative value would turn off this cut in case of 'custom' discriminator");
  desc_wp.add<bool>("doCaloMuonVeto");
  desc_wp.add<int>("maxNumberOfHitsLast2Stations")
      ->setComment("negative value would turn off this cut in case of 'custom' discriminator");
  edm::ParameterSet pset_wp;
  pset_wp.addParameter<std::string>("IDname", "pfRecoTauDiscriminationAgainstMuon2Container");
  pset_wp.addParameter<std::string>("discriminatorOption", "loose");
  pset_wp.addParameter<double>("HoPMin", 0.2);
  pset_wp.addParameter<int>("maxNumberOfMatches", 0);
  pset_wp.addParameter<bool>("doCaloMuonVeto", false);
  pset_wp.addParameter<int>("maxNumberOfHitsLast2Stations", 0);
  std::vector<edm::ParameterSet> vpsd_wp;
  vpsd_wp.push_back(pset_wp);
  desc.addVPSet("IDWPdefinitions", desc_wp, vpsd_wp);
  //add empty raw value config to simplify subsequent provenance searches
  desc.addVPSet("IDdefinitions", edm::ParameterSetDescription(), {});

  descriptions.add("pfRecoTauDiscriminationAgainstMuon2Container", desc);
}

DEFINE_FWK_MODULE(PFRecoTauDiscriminationAgainstMuon2Container);

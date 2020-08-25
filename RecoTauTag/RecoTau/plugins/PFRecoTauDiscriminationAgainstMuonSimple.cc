/** \class PFRecoTauDiscriminationAgainstMuonSimple
 *
 * Compute tau Id. discriminator against muons for MiniAOD.
 * 
 * \author Michal Bluj, NCBJ Warsaw
 * based on PFRecoTauDiscriminationAgainstMuon2 by Christian Veelken
 *
 * Note: it is not granted that information on muon track is/will be always 
 * accesible with MiniAOD, if not one can consider to veto muons which are 
 * not only Trk by also STA or RPC muons, i.e. have many (>=2) good muon segments 
 */

#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "RecoTauTag/RecoTau/interface/RecoTauMuonTools.h"

#include <vector>
#include <string>
#include <iostream>
#include <atomic>

using reco::tau::format_vint;

namespace {

  struct PFRecoTauDiscriminationAgainstMuonSimpleConfigSet {
    PFRecoTauDiscriminationAgainstMuonSimpleConfigSet(double hop, int mNOM, bool doCMV, int mNHL2S, int mNSTA, int mNRPC)
        : hop(hop),
          maxNumberOfMatches(mNOM),
          doCaloMuonVeto(doCMV),
          maxNumberOfHitsLast2Stations(mNHL2S),
          maxNumberOfSTAMuons(mNSTA),
          maxNumberOfRPCMuons(mNRPC) {}

    double hop;
    int maxNumberOfMatches;
    bool doCaloMuonVeto;
    int maxNumberOfHitsLast2Stations;
    int maxNumberOfSTAMuons, maxNumberOfRPCMuons;
  };

  class PFRecoTauDiscriminationAgainstMuonSimple final : public PFTauDiscriminationContainerProducerBase {
  public:
    explicit PFRecoTauDiscriminationAgainstMuonSimple(const edm::ParameterSet& cfg)
        : PFTauDiscriminationContainerProducerBase(cfg), moduleLabel_(cfg.getParameter<std::string>("@module_label")) {
      auto const wpDefs = cfg.getParameter<std::vector<edm::ParameterSet>>("IDWPdefinitions");
      for (auto& wpDef : wpDefs) {
        wpDefs_.push_back(
            PFRecoTauDiscriminationAgainstMuonSimpleConfigSet(wpDef.getParameter<double>("HoPMin"),
                                                              wpDef.getParameter<int>("maxNumberOfMatches"),
                                                              wpDef.getParameter<bool>("doCaloMuonVeto"),
                                                              wpDef.getParameter<int>("maxNumberOfHitsLast2Stations"),
                                                              wpDef.getParameter<int>("maxNumberOfSTAMuons"),
                                                              wpDef.getParameter<int>("maxNumberOfRPCMuons")));
      }
      srcPatMuons_ = cfg.getParameter<edm::InputTag>("srcPatMuons");
      muons_token = consumes<pat::MuonCollection>(srcPatMuons_);
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
    ~PFRecoTauDiscriminationAgainstMuonSimple() override {}

    void beginEvent(const edm::Event&, const edm::EventSetup&) override;

    reco::SingleTauDiscriminatorContainer discriminate(const reco::PFTauRef&) const override;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    std::string moduleLabel_;
    std::vector<PFRecoTauDiscriminationAgainstMuonSimpleConfigSet> wpDefs_;
    edm::InputTag srcPatMuons_;
    edm::Handle<pat::MuonCollection> muons_;
    edm::EDGetTokenT<pat::MuonCollection> muons_token;
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

  std::atomic<unsigned int> PFRecoTauDiscriminationAgainstMuonSimple::numWarnings_{0};

  void PFRecoTauDiscriminationAgainstMuonSimple::beginEvent(const edm::Event& evt, const edm::EventSetup& es) {
    evt.getByToken(muons_token, muons_);
  }

  reco::SingleTauDiscriminatorContainer PFRecoTauDiscriminationAgainstMuonSimple::discriminate(
      const reco::PFTauRef& pfTau) const {
    if (verbosity_) {
      edm::LogPrint("PFTauAgainstMuonSimple") << "<PFRecoTauDiscriminationAgainstMuonSimple::discriminate>:";
      edm::LogPrint("PFTauAgainstMuonSimple") << " moduleLabel = " << moduleLabel_;
      edm::LogPrint("PFTauAgainstMuonSimple")
          << "tau #" << pfTau.key() << ": Pt = " << pfTau->pt() << ", eta = " << pfTau->eta()
          << ", phi = " << pfTau->phi() << ", decay mode = " << pfTau->decayMode();
    }

    //energy fraction carried by leading charged hadron
    const reco::CandidatePtr& pfLeadChargedHadron = pfTau->leadChargedHadrCand();
    double caloEnergyFraction = 99;
    if (pfLeadChargedHadron.isNonnull()) {
      const pat::PackedCandidate* pCand = dynamic_cast<const pat::PackedCandidate*>(pfLeadChargedHadron.get());
      if (pCand != nullptr) {
        caloEnergyFraction = pCand->caloFraction();
        if (pCand->hasTrackDetails())  //MB: relate energy fraction to track momentum rather than to energy of candidate
          caloEnergyFraction *= pCand->energy() / pCand->bestTrack()->p();
        if (verbosity_) {
          edm::LogPrint("PFTauAgainstMuonSimple")
              << "decayMode = " << pfTau->decayMode() << ", caloEnergy(ECAL+HCAL)Fraction = " << caloEnergyFraction
              << ", leadPFChargedHadronP = " << pCand->p() << ", leadPFChargedHadron pdgId = " << pCand->pdgId();
        }
      }
    }
    //iterate over muons to find ones to use for discrimination
    std::vector<const pat::Muon*> muonsToCheck;
    size_t numMuons = muons_->size();
    for (size_t idxMuon = 0; idxMuon < numMuons; ++idxMuon) {
      bool matched = false;
      const pat::MuonRef muon(muons_, idxMuon);
      if (verbosity_)
        edm::LogPrint("PFTauAgainstMuonSimple") << "muon #" << muon.key() << ": Pt = " << muon->pt()
                                                << ", eta = " << muon->eta() << ", phi = " << muon->phi();
      //firsty check if the muon corrsponds with the leading tau particle
      for (size_t iCand = 0; iCand < muon->numberOfSourceCandidatePtrs(); ++iCand) {
        const reco::CandidatePtr& srcCand = muon->sourceCandidatePtr(iCand);
        if (srcCand.isNonnull() && pfLeadChargedHadron.isNonnull() && srcCand.id() == pfLeadChargedHadron.id() &&
            srcCand.key() == pfLeadChargedHadron.key()) {
          muonsToCheck.push_back(muon.get());
          matched = true;
          if (verbosity_)
            edm::LogPrint("PFTauAgainstMuonSimple") << " tau has muonRef.";
          break;
        }
      }
      if (matched)
        continue;
      //check dR matching
      if (!(muon->pt() > minPtMatchedMuon_)) {
        if (verbosity_) {
          edm::LogPrint("PFTauAgainstMuonSimple") << " fails Pt cut --> skipping it.";
        }
        continue;
      }
      double dR = deltaR(muon->p4(), pfTau->p4());
      double dRmatch = dRmuonMatch_;
      if (dRmuonMatchLimitedToJetArea_) {
        double jetArea = 0.;
        if (pfTau->jetRef().isNonnull())
          jetArea = pfTau->jetRef()->jetArea();
        if (jetArea > 0.) {
          dRmatch = std::min(dRmatch, std::sqrt(jetArea / M_PI));
        } else {
          if (numWarnings_ < maxWarnings_) {
            edm::LogInfo("PFRecoTauDiscriminationAgainstMuonSimple::discriminate")
                << "Jet associated to Tau: Pt = " << pfTau->pt() << ", eta = " << pfTau->eta()
                << ", phi = " << pfTau->phi() << " has area = " << jetArea << " !!";
            ++numWarnings_;
          }
          dRmatch = pfTau->signalConeSize();
        }
        dRmatch = std::max(dRmatch, pfTau->signalConeSize());
        if (dR < dRmatch) {
          if (verbosity_)
            edm::LogPrint("PFTauAgainstMuonSimple") << " overlaps with tau, dR = " << dR;
          muonsToCheck.push_back(muon.get());
        }
      }
    }
    //now examine selected muons
    std::vector<int> numMatchesDT(4);
    std::vector<int> numMatchesCSC(4);
    std::vector<int> numMatchesRPC(4);
    std::vector<int> numHitsDT(4);
    std::vector<int> numHitsCSC(4);
    std::vector<int> numHitsRPC(4);
    //MB: clear counters of matched segments and hits globally as in the AgainstMuon2 discriminant, but note that they will sum for all matched muons. However it is not likely that there is more than one matched muon.
    for (int iStation = 0; iStation < 4; ++iStation) {
      numMatchesDT[iStation] = 0;
      numMatchesCSC[iStation] = 0;
      numMatchesRPC[iStation] = 0;
      numHitsDT[iStation] = 0;
      numHitsCSC[iStation] = 0;
      numHitsRPC[iStation] = 0;
    }
    int numSTAMuons = 0, numRPCMuons = 0;
    int numStationsWithMatches = 0;
    int numLast2StationsWithHits = 0;
    if (verbosity_ && !muonsToCheck.empty())
      edm::LogPrint("PFTauAgainstMuonSimple") << "Muons to check (" << muonsToCheck.size() << "):";
    size_t iMu = 0;
    for (const auto& mu : muonsToCheck) {
      if (mu->isStandAloneMuon())
        numSTAMuons++;
      if (mu->muonID("RPCMuLoose"))
        numRPCMuons++;
      reco::tau::countMatches(*mu, numMatchesDT, numMatchesCSC, numMatchesRPC);
      for (int iStation = 0; iStation < 4; ++iStation) {
        if (numMatchesDT[iStation] > 0 && !maskMatchesDT_[iStation])
          ++numStationsWithMatches;
        if (numMatchesCSC[iStation] > 0 && !maskMatchesCSC_[iStation])
          ++numStationsWithMatches;
        if (numMatchesRPC[iStation] > 0 && !maskMatchesRPC_[iStation])
          ++numStationsWithMatches;
      }
      reco::tau::countHits(*mu, numHitsDT, numHitsCSC, numHitsRPC);
      for (int iStation = 2; iStation < 4; ++iStation) {
        if (numHitsDT[iStation] > 0 && !maskHitsDT_[iStation])
          ++numLast2StationsWithHits;
        if (numHitsCSC[iStation] > 0 && !maskHitsCSC_[iStation])
          ++numLast2StationsWithHits;
        if (numHitsRPC[iStation] > 0 && !maskHitsRPC_[iStation])
          ++numLast2StationsWithHits;
      }
      if (verbosity_)
        edm::LogPrint("PFTauAgainstMuonSimple")
            << "\t" << iMu << ": Pt = " << mu->pt() << ", eta = " << mu->eta() << ", phi = " << mu->phi() << "\n\t"
            << "   isSTA: " << mu->isStandAloneMuon() << ", isRPCLoose: " << mu->muonID("RPCMuLoose")
            << "\n\t   numMatchesDT  = " << format_vint(numMatchesDT)
            << "\n\t   numMatchesCSC = " << format_vint(numMatchesCSC)
            << "\n\t   numMatchesRPC = " << format_vint(numMatchesRPC)
            << "\n\t   --> numStationsWithMatches = " << numStationsWithMatches
            << "\n\t   numHitsDT  = " << format_vint(numHitsDT) << "\n\t   numHitsCSC = " << format_vint(numHitsCSC)
            << "\n\t   numHitsRPC = " << format_vint(numHitsRPC)
            << "\n\t   --> numLast2StationsWithHits = " << numLast2StationsWithHits;
      iMu++;
    }

    reco::SingleTauDiscriminatorContainer result;
    for (auto const& wpDef : wpDefs_) {
      bool discriminatorValue = true;
      if (wpDef.maxNumberOfMatches >= 0 && numStationsWithMatches > wpDef.maxNumberOfMatches)
        discriminatorValue = false;
      if (wpDef.maxNumberOfHitsLast2Stations >= 0 && numLast2StationsWithHits > wpDef.maxNumberOfHitsLast2Stations)
        discriminatorValue = false;
      if (wpDef.maxNumberOfSTAMuons >= 0 && numSTAMuons > wpDef.maxNumberOfSTAMuons) {
        discriminatorValue = false;
      }
      if (wpDef.maxNumberOfRPCMuons >= 0 && numRPCMuons > wpDef.maxNumberOfRPCMuons) {
        discriminatorValue = false;
      }
      bool passesCaloMuonVeto = true;
      if (pfTau->decayMode() == 0 && caloEnergyFraction < wpDef.hop) {
        passesCaloMuonVeto = false;
      }
      if (wpDef.doCaloMuonVeto && !passesCaloMuonVeto) {
        discriminatorValue = false;
      }
      result.workingPoints.push_back(discriminatorValue);
      if (verbosity_)
        edm::LogPrint("PFTauAgainstMuonSimple") << "--> returning discriminatorValue = " << result.workingPoints.back();
    }
    return result;
  }

  void PFRecoTauDiscriminationAgainstMuonSimple::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    // pfRecoTauDiscriminationAgainstMuonSimple
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("PFTauProducer", edm::InputTag("pfRecoTauProducer"));
    {
      edm::ParameterSetDescription psd0;
      psd0.add<std::string>("BooleanOperator", "and");
      {
        edm::ParameterSetDescription psd1;
        psd1.add<double>("cut");              //, 0.5);
        psd1.add<edm::InputTag>("Producer");  //, edm::InputTag("pfRecoTauDiscriminationByLeadingTrackFinding"));
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
    {
      edm::ParameterSetDescription desc_wp;
      desc_wp.add<std::string>("IDname");
      desc_wp.add<double>("HoPMin");
      desc_wp.add<int>("maxNumberOfMatches")->setComment("negative value would turn off this cut");
      desc_wp.add<bool>("doCaloMuonVeto");
      desc_wp.add<int>("maxNumberOfHitsLast2Stations")->setComment("negative value would turn off this cut");
      desc_wp.add<int>("maxNumberOfSTAMuons")->setComment("negative value would turn off this cut");
      desc_wp.add<int>("maxNumberOfRPCMuons")->setComment("negative value would turn off this cut");
      edm::ParameterSet pset_wp;
      pset_wp.addParameter<std::string>("IDname", "ByLooseMuonRejectionSimple");
      pset_wp.addParameter<double>("HoPMin", 0.2);
      pset_wp.addParameter<int>("maxNumberOfMatches", 1);
      pset_wp.addParameter<bool>("doCaloMuonVeto", true);
      pset_wp.addParameter<int>("maxNumberOfHitsLast2Stations", -1);
      pset_wp.addParameter<int>("maxNumberOfSTAMuons", -1);
      pset_wp.addParameter<int>("maxNumberOfRPCMuons", -1);
      std::vector<edm::ParameterSet> vpsd_wp;
      vpsd_wp.push_back(pset_wp);
      desc.addVPSet("IDWPdefinitions", desc_wp, vpsd_wp);
    }
    //add empty raw value config to simplify subsequent provenance searches
    desc.addVPSet("IDdefinitions", edm::ParameterSetDescription(), {});

    desc.add<edm::InputTag>("srcPatMuons", edm::InputTag("slimmedMuons"));
    desc.add<double>("dRmuonMatch", 0.3);
    desc.add<bool>("dRmuonMatchLimitedToJetArea", false);
    desc.add<double>("minPtMatchedMuon", 5.0);
    desc.add<std::vector<int>>("maskMatchesDT", {0, 0, 0, 0});
    desc.add<std::vector<int>>("maskMatchesCSC", {1, 0, 0, 0})
        ->setComment(
            "flags to mask/unmask DT, CSC and RPC chambers in individual muon stations. Segments and hits that are "
            "present in that muon station are ignored in case the 'mask' is set to 1. Per default only the innermost "
            "CSC "
            "chamber is ignored, as it is affected by spurious hits in high pile-up events.");
    desc.add<std::vector<int>>("maskMatchesRPC", {0, 0, 0, 0});
    desc.add<std::vector<int>>("maskHitsDT", {0, 0, 0, 0});
    desc.add<std::vector<int>>("maskHitsCSC", {0, 0, 0, 0});
    desc.add<std::vector<int>>("maskHitsRPC", {0, 0, 0, 0});
    desc.add<int>("verbosity", 0);
    descriptions.add("pfRecoTauDiscriminationAgainstMuonSimple", desc);
  }

}  // namespace

DEFINE_FWK_MODULE(PFRecoTauDiscriminationAgainstMuonSimple);

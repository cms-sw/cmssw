
/** \class PFRecoTauDiscriminationAgainstMuonMVA
 *
 * MVA based discriminator against muon -> tau fakes
 * 
 * \author Christian Veelken, LLR
 *
 */

#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "FWCore/Utilities/interface/Exception.h"

#include <FWCore/ParameterSet/interface/ConfigurationDescriptions.h>
#include <FWCore/ParameterSet/interface/ParameterSetDescription.h>

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "CondFormats/GBRForest/interface/GBRForest.h"
#include "CondFormats/DataRecord/interface/GBRWrapperRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include <TMath.h>
#include <TFile.h>

#include <iostream>

using namespace reco;

namespace {
  const GBRForest* loadMVAfromFile(const edm::FileInPath& inputFileName,
                                   const std::string& mvaName,
                                   std::vector<TFile*>& inputFilesToDelete) {
    if (inputFileName.location() == edm::FileInPath::Unknown)
      throw cms::Exception("PFRecoTauDiscriminationAgainstMuonMVA::loadMVA")
          << " Failed to find File = " << inputFileName << " !!\n";
    TFile* inputFile = new TFile(inputFileName.fullPath().data());

    //const GBRForest* mva = dynamic_cast<GBRForest*>(inputFile->Get(mvaName.data())); // CV: dynamic_cast<GBRForest*> fails for some reason ?!
    const GBRForest* mva = (GBRForest*)inputFile->Get(mvaName.data());
    if (!mva)
      throw cms::Exception("PFRecoTauDiscriminationAgainstMuonMVA::loadMVA")
          << " Failed to load MVA = " << mvaName.data() << " from file = " << inputFileName.fullPath().data()
          << " !!\n";

    inputFilesToDelete.push_back(inputFile);

    return mva;
  }

  const GBRForest* loadMVAfromDB(const edm::EventSetup& es, const std::string& mvaName) {
    edm::ESHandle<GBRForest> mva;
    es.get<GBRWrapperRcd>().get(mvaName, mva);
    return mva.product();
  }

  class PFRecoTauDiscriminationAgainstMuonMVA final : public PFTauDiscriminationContainerProducerBase {
  public:
    explicit PFRecoTauDiscriminationAgainstMuonMVA(const edm::ParameterSet& cfg)
        : PFTauDiscriminationContainerProducerBase(cfg),
          moduleLabel_(cfg.getParameter<std::string>("@module_label")),
          mvaReader_(nullptr),
          mvaInput_(nullptr) {
      mvaName_ = cfg.getParameter<std::string>("mvaName");
      loadMVAfromDB_ = cfg.getParameter<bool>("loadMVAfromDB");
      if (!loadMVAfromDB_) {
        inputFileName_ = cfg.getParameter<edm::FileInPath>("inputFileName");
      }
      mvaInput_ = new float[11];

      srcMuons_ = cfg.getParameter<edm::InputTag>("srcMuons");
      Muons_token = consumes<reco::MuonCollection>(srcMuons_);
      dRmuonMatch_ = cfg.getParameter<double>("dRmuonMatch");

      verbosity_ = cfg.getParameter<int>("verbosity");
    }

    void beginEvent(const edm::Event&, const edm::EventSetup&) override;

    reco::SingleTauDiscriminatorContainer discriminate(const PFTauRef&) const override;

    ~PFRecoTauDiscriminationAgainstMuonMVA() override {
      if (!loadMVAfromDB_)
        delete mvaReader_;
      delete[] mvaInput_;
      for (std::vector<TFile*>::iterator it = inputFilesToDelete_.begin(); it != inputFilesToDelete_.end(); ++it) {
        delete (*it);
      }
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    std::string moduleLabel_;

    std::string mvaName_;
    bool loadMVAfromDB_;
    edm::FileInPath inputFileName_;
    const GBRForest* mvaReader_;
    float* mvaInput_;

    edm::InputTag srcMuons_;
    edm::Handle<reco::MuonCollection> muons_;
    edm::EDGetTokenT<reco::MuonCollection> Muons_token;
    double dRmuonMatch_;

    edm::Handle<TauCollection> taus_;

    std::vector<TFile*> inputFilesToDelete_;

    int verbosity_;
  };

  void PFRecoTauDiscriminationAgainstMuonMVA::beginEvent(const edm::Event& evt, const edm::EventSetup& es) {
    if (!mvaReader_) {
      if (loadMVAfromDB_) {
        mvaReader_ = loadMVAfromDB(es, mvaName_);
      } else {
        mvaReader_ = loadMVAfromFile(inputFileName_, mvaName_, inputFilesToDelete_);
      }
    }

    evt.getByToken(Muons_token, muons_);

    evt.getByToken(Tau_token, taus_);
  }

  namespace {
    void countHits(const reco::Muon& muon,
                   std::vector<int>& numHitsDT,
                   std::vector<int>& numHitsCSC,
                   std::vector<int>& numHitsRPC) {
      if (muon.outerTrack().isNonnull()) {
        const reco::HitPattern& muonHitPattern = muon.outerTrack()->hitPattern();
        for (int iHit = 0; iHit < muonHitPattern.numberOfAllHits(HitPattern::TRACK_HITS); ++iHit) {
          uint32_t hit = muonHitPattern.getHitPattern(HitPattern::TRACK_HITS, iHit);
          if (hit == 0)
            break;
          if (muonHitPattern.muonHitFilter(hit) && (muonHitPattern.getHitType(hit) == TrackingRecHit::valid ||
                                                    muonHitPattern.getHitType(hit) == TrackingRecHit::bad)) {
            int muonStation = muonHitPattern.getMuonStation(hit) - 1;  // CV: map into range 0..3
            if (muonStation >= 0 && muonStation < 4) {
              if (muonHitPattern.muonDTHitFilter(hit))
                ++numHitsDT[muonStation];
              else if (muonHitPattern.muonCSCHitFilter(hit))
                ++numHitsCSC[muonStation];
              else if (muonHitPattern.muonRPCHitFilter(hit))
                ++numHitsRPC[muonStation];
            }
          }
        }
      }
    }
  }  // namespace

  reco::SingleTauDiscriminatorContainer PFRecoTauDiscriminationAgainstMuonMVA::discriminate(const PFTauRef& tau) const {
    if (verbosity_) {
      edm::LogPrint("PFTauAgainstMuonMVA") << "<PFRecoTauDiscriminationAgainstMuonMVA::discriminate>:";
      edm::LogPrint("PFTauAgainstMuonMVA") << " moduleLabel = " << moduleLabel_;
    }

    reco::SingleTauDiscriminatorContainer result;
    // CV: define dummy category index in order to use RecoTauDiscriminantCutMultiplexer module to apply WP cuts
    result.rawValues = {-1., 0.};

    // CV: computation of anti-muon MVA value requires presence of leading charged hadron
    if (tau->leadPFChargedHadrCand().isNull())
      return 0.;

    mvaInput_[0] = TMath::Abs(tau->eta());
    double tauCaloEnECAL = 0.;
    double tauCaloEnHCAL = 0.;
    const std::vector<reco::PFCandidatePtr>& tauSignalPFCands = tau->signalPFCands();
    for (std::vector<reco::PFCandidatePtr>::const_iterator tauSignalPFCand = tauSignalPFCands.begin();
         tauSignalPFCand != tauSignalPFCands.end();
         ++tauSignalPFCand) {
      tauCaloEnECAL += (*tauSignalPFCand)->ecalEnergy();
      tauCaloEnHCAL += (*tauSignalPFCand)->hcalEnergy();
    }
    mvaInput_[1] = TMath::Sqrt(TMath::Max(0., tauCaloEnECAL));
    mvaInput_[2] = TMath::Sqrt(TMath::Max(0., tauCaloEnHCAL));
    mvaInput_[3] = tau->leadPFChargedHadrCand()->pt() / TMath::Max(1., Double_t(tau->pt()));
    mvaInput_[4] = TMath::Sqrt(TMath::Max(0., tau->leadPFChargedHadrCand()->ecalEnergy()));
    mvaInput_[5] = TMath::Sqrt(TMath::Max(0., tau->leadPFChargedHadrCand()->hcalEnergy()));
    int numMatches = 0;
    std::vector<int> numHitsDT(4);
    std::vector<int> numHitsCSC(4);
    std::vector<int> numHitsRPC(4);
    for (int iStation = 0; iStation < 4; ++iStation) {
      numHitsDT[iStation] = 0;
      numHitsCSC[iStation] = 0;
      numHitsRPC[iStation] = 0;
    }
    if (tau->leadPFChargedHadrCand().isNonnull()) {
      reco::MuonRef muonRef = tau->leadPFChargedHadrCand()->muonRef();
      if (muonRef.isNonnull()) {
        numMatches = muonRef->numberOfMatches(reco::Muon::NoArbitration);
        countHits(*muonRef, numHitsDT, numHitsCSC, numHitsRPC);
      }
    }
    size_t numMuons = muons_->size();
    for (size_t idxMuon = 0; idxMuon < numMuons; ++idxMuon) {
      reco::MuonRef muon(muons_, idxMuon);
      if (tau->leadPFChargedHadrCand().isNonnull() && tau->leadPFChargedHadrCand()->muonRef().isNonnull() &&
          muon == tau->leadPFChargedHadrCand()->muonRef()) {
        continue;
      }
      double dR = deltaR(muon->p4(), tau->p4());
      if (dR < dRmuonMatch_) {
        numMatches += muon->numberOfMatches(reco::Muon::NoArbitration);
        countHits(*muon, numHitsDT, numHitsCSC, numHitsRPC);
      }
    }
    mvaInput_[6] = numMatches;
    mvaInput_[7] = numHitsDT[0] + numHitsCSC[0] + numHitsRPC[0];
    mvaInput_[8] = numHitsDT[1] + numHitsCSC[1] + numHitsRPC[1];
    mvaInput_[9] = numHitsDT[2] + numHitsCSC[2] + numHitsRPC[2];
    mvaInput_[10] = numHitsDT[3] + numHitsCSC[3] + numHitsRPC[3];

    result.rawValues.at(0) = mvaReader_->GetClassifier(mvaInput_);
    if (verbosity_) {
      edm::LogPrint("PFTauAgainstMuonMVA") << "mvaValue = " << result.rawValues.at(0);
    }
    return result.rawValues.at(0);
  }
}  // namespace

void PFRecoTauDiscriminationAgainstMuonMVA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // pfRecoTauDiscriminationAgainstMuonMVA
  edm::ParameterSetDescription desc;
  desc.add<double>("mvaMin", 0.0);
  desc.add<std::string>("mvaName", "againstMuonMVA");
  desc.add<edm::InputTag>("PFTauProducer", edm::InputTag("pfTauProducer"));
  desc.add<int>("verbosity", 0);
  desc.add<bool>("returnMVA", true);
  desc.add<edm::FileInPath>("inputFileName", edm::FileInPath("RecoTauTag/RecoTau/data/emptyMVAinputFile"));
  desc.add<bool>("loadMVAfromDB", true);
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("BooleanOperator", "and");
    {
      edm::ParameterSetDescription psd1;
      psd1.add<double>("cut");
      psd1.add<edm::InputTag>("Producer");
      psd0.addOptional<edm::ParameterSetDescription>("leadTrack", psd1);
    }
    desc.add<edm::ParameterSetDescription>("Prediscriminants", psd0);
  }
  desc.add<double>("dRmuonMatch", 0.3);
  desc.add<edm::InputTag>("srcMuons", edm::InputTag("muons"));
  descriptions.add("pfRecoTauDiscriminationAgainstMuonMVA", desc);
}

DEFINE_FWK_MODULE(PFRecoTauDiscriminationAgainstMuonMVA);

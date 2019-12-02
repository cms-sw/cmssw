#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementSuperClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementSuperCluster.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "RecoParticleFlow/PFProducer/interface/PFEGammaFilters.h"
#include "RecoParticleFlow/PFProducer/interface/PFAlgo.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyCalibration.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyCalibrationHF.h"
#include "CondFormats/PhysicsToolsObjects/interface/PerformancePayloadFromTFormula.h"
#include "CondFormats/DataRecord/interface/PFCalibrationRcd.h"
#include "CondFormats/DataRecord/interface/GBRWrapperRcd.h"

#include <sstream>
#include <string>

#include "TFile.h"

/**\class PFProducer
\brief Producer for particle flow reconstructed particles (PFCandidates)

This producer makes use of PFAlgo, the particle flow algorithm.

\author Colin Bernet
\date   July 2006
*/

class PFProducer : public edm::stream::EDProducer<> {
public:
  explicit PFProducer(const edm::ParameterSet&);

  void produce(edm::Event&, const edm::EventSetup&) override;
  void beginRun(const edm::Run&, const edm::EventSetup&) override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  const edm::EDPutTokenT<reco::PFCandidateCollection> pfCandidatesToken_;
  const edm::EDPutTokenT<reco::PFCandidateCollection> pfCleanedCandidatesToken_;

  const edm::EDGetTokenT<reco::PFBlockCollection> inputTagBlocks_;
  edm::EDGetTokenT<reco::MuonCollection> inputTagMuons_;
  edm::EDGetTokenT<reco::VertexCollection> vertices_;
  edm::EDGetTokenT<reco::GsfElectronCollection> inputTagEgammaElectrons_;

  std::vector<edm::EDGetTokenT<reco::PFRecHitCollection>> inputTagCleanedHF_;
  std::string electronExtraOutputCol_;
  std::string photonExtraOutputCol_;

  // NEW EGamma Filters
  edm::EDGetTokenT<edm::ValueMap<reco::GsfElectronRef>> inputTagValueMapGedElectrons_;
  edm::EDGetTokenT<edm::ValueMap<reco::PhotonRef>> inputTagValueMapGedPhotons_;
  edm::EDGetTokenT<edm::View<reco::PFCandidate>> inputTagPFEGammaCandidates_;

  bool use_EGammaFilters_;
  std::unique_ptr<PFEGammaFilters> pfegamma_ = nullptr;

  //Use of HO clusters and links in PF Reconstruction
  bool useHO_;

  /// verbose ?
  bool verbose_;

  // Post muon cleaning ?
  bool postMuonCleaning_;

  // what about e/g electrons ?
  bool useEGammaElectrons_;

  // Use vertices for Neutral particles ?
  bool useVerticesForNeutral_;

  // Take PF cluster calibrations from Global Tag ?
  bool useCalibrationsFromDB_;
  std::string calibrationsLabel_;

  bool postHFCleaning_;
  // Name of the calibration functions to read from the database
  // std::vector<std::string> fToRead;

  // calibrations
  PFEnergyCalibration pfEnergyCalibration_;
  PFEnergyCalibrationHF pfEnergyCalibrationHF_;

  /// particle flow algorithm
  PFAlgo pfAlgo_;
};

DEFINE_FWK_MODULE(PFProducer);

using namespace std;
using namespace edm;

PFProducer::PFProducer(const edm::ParameterSet& iConfig)
    : pfCandidatesToken_{produces<reco::PFCandidateCollection>()},
      pfCleanedCandidatesToken_{produces<reco::PFCandidateCollection>("CleanedHF")},
      inputTagBlocks_(consumes<reco::PFBlockCollection>(iConfig.getParameter<InputTag>("blocks"))),
      pfEnergyCalibrationHF_(iConfig.getParameter<bool>("calibHF_use"),
                             iConfig.getParameter<std::vector<double>>("calibHF_eta_step"),
                             iConfig.getParameter<std::vector<double>>("calibHF_a_EMonly"),
                             iConfig.getParameter<std::vector<double>>("calibHF_b_HADonly"),
                             iConfig.getParameter<std::vector<double>>("calibHF_a_EMHAD"),
                             iConfig.getParameter<std::vector<double>>("calibHF_b_EMHAD")),
      pfAlgo_(iConfig.getParameter<double>("pf_nsigma_ECAL"),
              iConfig.getParameter<double>("pf_nsigma_HCAL"),
              iConfig.getParameter<double>("pf_nsigma_HFEM"),
              iConfig.getParameter<double>("pf_nsigma_HFHAD"),
              iConfig.getParameter<std::vector<double>>("resolHF_square"),
              pfEnergyCalibration_,
              pfEnergyCalibrationHF_,
              iConfig) {
  //Post cleaning of the muons
  inputTagMuons_ = consumes<reco::MuonCollection>(iConfig.getParameter<InputTag>("muons"));
  postMuonCleaning_ = iConfig.getParameter<bool>("postMuonCleaning");

  if (iConfig.existsAs<bool>("useEGammaFilters")) {
    use_EGammaFilters_ = iConfig.getParameter<bool>("useEGammaFilters");
  } else {
    use_EGammaFilters_ = false;
  }

  useEGammaElectrons_ = iConfig.getParameter<bool>("useEGammaElectrons");

  if (useEGammaElectrons_) {
    inputTagEgammaElectrons_ =
        consumes<reco::GsfElectronCollection>(iConfig.getParameter<edm::InputTag>("egammaElectrons"));
  }

  // register products
  produces<reco::PFCandidateCollection>("CleanedCosmicsMuons");
  produces<reco::PFCandidateCollection>("CleanedTrackerAndGlobalMuons");
  produces<reco::PFCandidateCollection>("CleanedFakeMuons");
  produces<reco::PFCandidateCollection>("CleanedPunchThroughMuons");
  produces<reco::PFCandidateCollection>("CleanedPunchThroughNeutralHadrons");
  produces<reco::PFCandidateCollection>("AddedMuonsAndHadrons");

  // Reading new EGamma selection cuts
  bool useProtectionsForJetMET(false);
  // Reading new EGamma ubiased collections and value maps
  if (use_EGammaFilters_) {
    inputTagPFEGammaCandidates_ =
        consumes<edm::View<reco::PFCandidate>>((iConfig.getParameter<edm::InputTag>("PFEGammaCandidates")));
    inputTagValueMapGedElectrons_ =
        consumes<edm::ValueMap<reco::GsfElectronRef>>(iConfig.getParameter<edm::InputTag>("GedElectronValueMap"));
    inputTagValueMapGedPhotons_ =
        consumes<edm::ValueMap<reco::PhotonRef>>(iConfig.getParameter<edm::InputTag>("GedPhotonValueMap"));
    useProtectionsForJetMET = iConfig.getParameter<bool>("useProtectionsForJetMET");
  }

  //Secondary tracks and displaced vertices parameters

  bool rejectTracks_Bad = iConfig.getParameter<bool>("rejectTracks_Bad");

  bool rejectTracks_Step45 = iConfig.getParameter<bool>("rejectTracks_Step45");

  bool usePFNuclearInteractions = iConfig.getParameter<bool>("usePFNuclearInteractions");

  bool usePFConversions = iConfig.getParameter<bool>("usePFConversions");

  bool usePFDecays = iConfig.getParameter<bool>("usePFDecays");

  double dptRel_DispVtx = iConfig.getParameter<double>("dptRel_DispVtx");

  useCalibrationsFromDB_ = iConfig.getParameter<bool>("useCalibrationsFromDB");

  if (useCalibrationsFromDB_)
    calibrationsLabel_ = iConfig.getParameter<std::string>("calibrationsLabel");

  // EGamma filters
  pfAlgo_.setEGammaParameters(use_EGammaFilters_, useProtectionsForJetMET);

  if (use_EGammaFilters_) {
    const edm::ParameterSet pfEGammaFiltersParams =
        iConfig.getParameter<edm::ParameterSet>("PFEGammaFiltersParameters");
    pfegamma_ = std::make_unique<PFEGammaFilters>(pfEGammaFiltersParams);
  }

  // Secondary tracks and displaced vertices parameters
  pfAlgo_.setDisplacedVerticesParameters(
      rejectTracks_Bad, rejectTracks_Step45, usePFNuclearInteractions, usePFConversions, usePFDecays, dptRel_DispVtx);

  if (usePFNuclearInteractions)
    pfAlgo_.setCandConnectorParameters(iConfig.getParameter<edm::ParameterSet>("iCfgCandConnector"));

  // Post cleaning of the HF
  postHFCleaning_ = iConfig.getParameter<bool>("postHFCleaning");
  const edm::ParameterSet pfHFCleaningParams = iConfig.getParameter<edm::ParameterSet>("PFHFCleaningParameters");

  // Set post HF cleaning muon parameters
  pfAlgo_.setPostHFCleaningParameters(postHFCleaning_, pfHFCleaningParams);

  // Input tags for HF cleaned rechits
  std::vector<edm::InputTag> tags = iConfig.getParameter<std::vector<edm::InputTag>>("cleanedHF");
  for (unsigned int i = 0; i < tags.size(); ++i)
    inputTagCleanedHF_.push_back(consumes<reco::PFRecHitCollection>(tags[i]));
  //MIKE: Vertex Parameters
  vertices_ = consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertexCollection"));
  useVerticesForNeutral_ = iConfig.getParameter<bool>("useVerticesForNeutral");

  // Use HO clusters and links in the PF reconstruction
  useHO_ = iConfig.getParameter<bool>("useHO");
  pfAlgo_.setHOTag(useHO_);

  verbose_ = iConfig.getUntrackedParameter<bool>("verbose", false);
}

void PFProducer::beginRun(const edm::Run& run, const edm::EventSetup& es) {
  if (useCalibrationsFromDB_) {
    // read the PFCalibration functions from the global tags
    edm::ESHandle<PerformancePayload> perfH;
    es.get<PFCalibrationRcd>().get(calibrationsLabel_, perfH);

    PerformancePayloadFromTFormula const* pfCalibrations =
        static_cast<const PerformancePayloadFromTFormula*>(perfH.product());

    pfEnergyCalibration_.setCalibrationFunctions(pfCalibrations);
  }
}

void PFProducer::produce(Event& iEvent, const EventSetup& iSetup) {
  LogDebug("PFProducer") << "START event: " << iEvent.id().event() << " in run " << iEvent.id().run() << endl;

  //Assign the PFAlgo Parameters
  pfAlgo_.setPFVertexParameters(useVerticesForNeutral_, iEvent.get(vertices_));

  // get the collection of blocks
  auto blocks = iEvent.getHandle(inputTagBlocks_);
  assert(blocks.isValid());

  // get the collection of muons
  if (postMuonCleaning_)
    pfAlgo_.setMuonHandle(iEvent.getHandle(inputTagMuons_));

  if (use_EGammaFilters_)
    pfAlgo_.setEGammaCollections(iEvent.get(inputTagPFEGammaCandidates_),
                                 iEvent.get(inputTagValueMapGedElectrons_),
                                 iEvent.get(inputTagValueMapGedPhotons_));

  LogDebug("PFProducer") << "particle flow is starting" << endl;

  pfAlgo_.reconstructParticles(blocks, pfegamma_.get());

  if (verbose_) {
    ostringstream str;
    str << pfAlgo_ << endl;
    LogInfo("PFProducer") << str.str() << endl;
  }

  // Check HF overcleaning
  if (postHFCleaning_) {
    reco::PFRecHitCollection hfCopy;
    for (unsigned ihf = 0; ihf < inputTagCleanedHF_.size(); ++ihf) {
      Handle<reco::PFRecHitCollection> hfCleaned;
      bool foundHF = iEvent.getByToken(inputTagCleanedHF_[ihf], hfCleaned);
      if (!foundHF)
        continue;
      for (unsigned jhf = 0; jhf < (*hfCleaned).size(); ++jhf) {
        hfCopy.push_back((*hfCleaned)[jhf]);
      }
    }
    pfAlgo_.checkCleaning(hfCopy);
  }

  // Save the final PFCandidate collection
  auto pOutputCandidateCollection = pfAlgo_.makeConnectedCandidates();

  LogDebug("PFProducer") << "particle flow: putting products in the event";
  if (verbose_) {
    int nC = 0;
    ostringstream ss;
    for (auto const& cand : pOutputCandidateCollection) {
      nC++;
      ss << "  " << nC << ") pid=" << cand.particleId() << " pt=" << cand.pt() << endl;
    }
    LogDebug("PFProducer") << "Here the full list:" << endl << ss.str();
  }

  // Write in the event
  iEvent.emplace(pfCandidatesToken_, pOutputCandidateCollection);
  iEvent.emplace(pfCleanedCandidatesToken_, pfAlgo_.getCleanedCandidates());

  if (postMuonCleaning_) {
    auto& muAlgo = *pfAlgo_.getPFMuonAlgo();
    // Save cosmic cleaned muon candidates
    iEvent.put(muAlgo.transferCleanedCosmicCandidates(), "CleanedCosmicsMuons");
    // Save tracker/global cleaned muon candidates
    iEvent.put(muAlgo.transferCleanedTrackerAndGlobalCandidates(), "CleanedTrackerAndGlobalMuons");
    // Save fake cleaned muon candidates
    iEvent.put(muAlgo.transferCleanedFakeCandidates(), "CleanedFakeMuons");
    // Save punch-through cleaned muon candidates
    iEvent.put(muAlgo.transferPunchThroughCleanedMuonCandidates(), "CleanedPunchThroughMuons");
    // Save punch-through cleaned neutral hadron candidates
    iEvent.put(muAlgo.transferPunchThroughCleanedHadronCandidates(), "CleanedPunchThroughNeutralHadrons");
    // Save added muon candidates
    iEvent.put(muAlgo.transferAddedMuonCandidates(), "AddedMuonsAndHadrons");
  }
}

void PFProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  // Verbosity and debug flags
  desc.addUntracked<bool>("verbose", false);
  desc.addUntracked<bool>("debug", false);

  // PF Blocks label
  desc.add<edm::InputTag>("blocks", edm::InputTag("particleFlowBlock"));

  // reco::muons label and Post Muon cleaning
  desc.add<edm::InputTag>("muons", edm::InputTag("muons1stStep"));
  desc.add<bool>("postMuonCleaning", true);

  // Vertices label
  desc.add<edm::InputTag>("vertexCollection", edm::InputTag("offlinePrimaryVertices"));
  desc.add<bool>("useVerticesForNeutral", true);

  // Use HO clusters in PF hadron reconstruction
  desc.add<bool>("useHO", true);

  // EGamma-related
  desc.add<edm::InputTag>("PFEGammaCandidates", edm::InputTag("particleFlowEGamma"));
  desc.add<edm::InputTag>("GedElectronValueMap", edm::InputTag("gedGsfElectronsTmp"));
  desc.add<edm::InputTag>("GedPhotonValueMap", edm::InputTag("gedPhotonsTmp", "valMapPFEgammaCandToPhoton"));

  desc.add<bool>("useEGammaElectrons", true);
  desc.add<edm::InputTag>("egammaElectrons", edm::InputTag("mvaElectrons"));

  desc.add<bool>("useEGammaFilters", true);
  desc.add<bool>("useProtectionsForJetMET", true);  // Propagated to PFEGammaFilters

  // For PFEGammaFilters
  edm::ParameterSetDescription psd_PFEGammaFilters;
  PFEGammaFilters::fillPSetDescription(psd_PFEGammaFilters);
  desc.add<edm::ParameterSetDescription>("PFEGammaFiltersParameters", psd_PFEGammaFilters);

  // Treatment of muons :
  // Expected energy in ECAL and HCAL, and RMS
  desc.add<std::vector<double>>("muon_HCAL", {3.0, 3.0});
  desc.add<std::vector<double>>("muon_ECAL", {0.5, 0.5});
  desc.add<std::vector<double>>("muon_HO", {0.9, 0.9});

  // For PFMuonAlgo
  edm::ParameterSetDescription psd_PFMuonAlgo;
  PFEGammaFilters::fillPSetDescription(psd_PFMuonAlgo);
  desc.add<edm::ParameterSetDescription>("PFMuonAlgoParameters", psd_PFMuonAlgo);

  // Input displaced vertices
  // It is strongly adviced to keep usePFNuclearInteractions = bCorrect
  desc.add<bool>("rejectTracks_Bad", true);
  desc.add<bool>("rejectTracks_Step45", true);

  desc.add<bool>("usePFNuclearInteractions", true);
  desc.add<bool>("usePFConversions", true);
  desc.add<bool>("usePFDecays", false);

  desc.add<double>("dptRel_DispVtx", 10.0);

  // PFCandConnector
  edm::ParameterSetDescription psd_CandConnector;
  PFCandConnector::fillPSetDescription(psd_CandConnector);
  desc.add<edm::ParameterSetDescription>("iCfgCandConnector", psd_CandConnector);

  // Treatment of potential fake tracks
  desc.add<double>("nsigma_TRACK", 1.0)->setComment("Number of sigmas for fake track detection");
  // pt_Error: dont forget to modify also ptErrorSecondary if you modify this parameter
  desc.add<double>("pt_Error", 1.0)
      ->setComment("Absolute pt error to detect fake tracks in the first three iterations");
  desc.add<std::vector<double>>("factors_45", {10.0, 100.0})
      ->setComment("Factors to be applied in the four and fifth steps to the pt error");

  // Treatment of tracks in region of bad HCal
  desc.add<double>("goodTrackDeadHcal_ptErrRel", 0.2)->setComment("trackRef->ptError()/trackRef->pt() < X");
  desc.add<double>("goodTrackDeadHcal_chi2n", 5)->setComment("trackRef->normalizedChi2() < X");
  desc.add<unsigned int>("goodTrackDeadHcal_layers", 4)
      ->setComment("trackRef->hitPattern().trackerLayersWithMeasurement() >= X");
  desc.add<double>("goodTrackDeadHcal_validFr", 0.5)->setComment("trackRef->validFraction() > X");
  desc.add<double>("goodTrackDeadHcal_dxy", 0.5)->setComment("abs(trackRef->dxy(primaryVertex_.position())) < X [cm]");

  desc.add<double>("goodPixelTrackDeadHcal_minEta", 2.3)->setComment("abs(trackRef->eta()) > X");
  desc.add<double>("goodPixelTrackDeadHcal_maxPt", 50.0)->setComment("trackRef->ptError()/trackRef->pt() < X");
  desc.add<double>("goodPixelTrackDeadHcal_ptErrRel", 1.0)->setComment("trackRef->ptError()/trackRef->pt() < X");
  desc.add<double>("goodPixelTrackDeadHcal_chi2n", 2)->setComment("trackRef->normalizedChi2() < X");
  desc.add<int>("goodPixelTrackDeadHcal_maxLost3Hit", 0)
      ->setComment(
          "max missing outer hits for a track with 3 valid pixel layers (can set to -1 to reject all these tracks)");
  desc.add<int>("goodPixelTrackDeadHcal_maxLost4Hit", 1)
      ->setComment("max missing outer hits for a track with >= 4 valid pixel layers");
  desc.add<double>("goodPixelTrackDeadHcal_dxy", 0.02)
      ->setComment("abs(trackRef->dxy(primaryVertex_.position())) < X [cm] ");
  desc.add<double>("goodPixelTrackDeadHcal_dz", 0.05)
      ->setComment("abs(trackRef->dz(primaryVertex_.position())) < X [cm]");

  // number of sigmas for neutral energy detection
  desc.add<double>("pf_nsigma_ECAL", 0.0);
  desc.add<double>("pf_nsigma_HCAL", 1.0);
  desc.add<double>("pf_nsigma_HFEM", 1.0);
  desc.add<double>("pf_nsigma_HFHAD", 1.0);

  // ECAL/HCAL PF cluster calibration : take it from global tag ?
  desc.add<bool>("useCalibrationsFromDB", true);
  desc.add<std::string>("calibrationsLabel", "");

  // Post HF cleaning
  desc.add<bool>("postHFCleaning", false);
  {
    edm::ParameterSetDescription psd_PFHFCleaning;
    psd_PFHFCleaning.add<double>("minHFCleaningPt", 5.0)
        ->setComment("Clean only objects with pt larger than this value");
    psd_PFHFCleaning.add<double>("maxSignificance", 2.5)
        ->setComment("Clean only if the initial MET/sqrt(sumet) is larger than this value");
    psd_PFHFCleaning.add<double>("minSignificance", 2.5)
        ->setComment("Clean only if the final MET/sqrt(sumet) is smaller than this value");
    psd_PFHFCleaning.add<double>("minSignificanceReduction", 1.4)
        ->setComment("Clean only if the significance reduction is larger than this value");
    psd_PFHFCleaning.add<double>("maxDeltaPhiPt", 7.0)
        ->setComment("Clean only if the MET and the to-be-cleaned object satisfy this DeltaPhi * Pt cut");
    // (the MET angular resoution is in 1/MET)
    psd_PFHFCleaning.add<double>("minDeltaMet", 0.4)
        ->setComment(
            "Clean only if the MET relative reduction from the to-be-cleaned object is larger than this value");
    desc.add<edm::ParameterSetDescription>("PFHFCleaningParameters", psd_PFHFCleaning);
  }

  // Check HF cleaning
  desc.add<std::vector<edm::InputTag>>("cleanedHF",
                                       {
                                           edm::InputTag("particleFlowRecHitHF", "Cleaned"),
                                           edm::InputTag("particleFlowClusterHF", "Cleaned"),
                                       });

  // calibration parameters for HF:
  desc.add<bool>("calibHF_use", false);
  desc.add<std::vector<double>>("calibHF_eta_step", {0.0, 2.9, 3.0, 3.2, 4.2, 4.4, 4.6, 4.8, 5.2, 5.4});
  desc.add<std::vector<double>>("calibHF_a_EMonly", {1., 1., 1., 1., 1., 1., 1., 1., 1., 1.});
  desc.add<std::vector<double>>("calibHF_a_EMHAD", {1., 1., 1., 1., 1., 1., 1., 1., 1., 1.});
  desc.add<std::vector<double>>("calibHF_b_HADonly", {1., 1., 1., 1., 1., 1., 1., 1., 1., 1.});
  desc.add<std::vector<double>>("calibHF_b_EMHAD", {1., 1., 1., 1., 1., 1., 1., 1., 1., 1.});

  // resolution parameters for HF: EPJC 53(2008)139, doi:10.1140/epjc/s10052-007-0459-4
  desc.add<std::vector<double>>("resolHF_square", {2.799 * 2.799, 0.114 * 0.114, 0.0 * 0.0})
      ->setComment("HF resolution - stochastic, constant, noise term squares");

  descriptions.add("particleFlow", desc);
}

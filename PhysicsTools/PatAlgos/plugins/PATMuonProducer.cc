/**
  \class    pat::PATMuonProducer PATMuonProducer.h "PhysicsTools/PatAlgos/interface/PATMuonProducer.h"
  \brief    Produces pat::Muon's

   The PATMuonProducer produces analysis-level pat::Muon's starting from
   a collection of objects of reco::Muon.

  \author   Steven Lowette, Roger Wolf
  \version  $Id: PATMuonProducer.h,v 1.29 2012/08/22 15:02:52 bellan Exp $
*/

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonSimInfo.h"
#include "DataFormats/MuonReco/interface/MuonTimeExtra.h"
#include "DataFormats/ParticleFlowCandidate/interface/IsolatedPFCandidate.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/PFIsolation.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/PatCandidates/interface/TriggerObjectStandAlone.h"
#include "DataFormats/PatCandidates/interface/UserData.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/EmptyGroupDescription.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/transform.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "JetMETCorrections/JetCorrector/interface/JetCorrector.h"
#include "PhysicsTools/PatAlgos/interface/EfficiencyLoader.h"
#include "PhysicsTools/PatAlgos/interface/KinResolutionsLoader.h"
#include "PhysicsTools/PatAlgos/interface/MultiIsolator.h"
#include "PhysicsTools/PatAlgos/interface/CalculatePtRatioRel.h"
#include "PhysicsTools/PatAlgos/interface/MuonMvaIDEstimator.h"
#include "PhysicsTools/PatAlgos/interface/PATUserDataHelper.h"
#include "PhysicsTools/PatAlgos/interface/SoftMuonMvaEstimator.h"
#include "PhysicsTools/PatUtils/interface/MiniIsolation.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"

namespace pat {

  class PATMuonHeavyObjectCache {
  public:
    PATMuonHeavyObjectCache(const edm::ParameterSet&);

    pat::CalculatePtRatioRel const& calculatePtRatioRel() const { return *calculatePtRatioRel_; }
    pat::MuonMvaIDEstimator const& muonMvaIDEstimator() const { return *muonMvaIDEstimator_; }
    pat::SoftMuonMvaEstimator const& softMuonMvaEstimator() const { return *softMuonMvaEstimator_; }

  private:
    std::unique_ptr<const pat::CalculatePtRatioRel> calculatePtRatioRel_;
    std::unique_ptr<const pat::MuonMvaIDEstimator> muonMvaIDEstimator_;
    std::unique_ptr<const pat::SoftMuonMvaEstimator> softMuonMvaEstimator_;
  };

  /// foward declarations
  class TrackerIsolationPt;
  class CaloIsolationEnergy;

  /// class definition
  class PATMuonProducer : public edm::stream::EDProducer<edm::GlobalCache<PATMuonHeavyObjectCache>> {
  public:
    /// default constructir
    explicit PATMuonProducer(const edm::ParameterSet& iConfig, PATMuonHeavyObjectCache const*);
    /// default destructur
    ~PATMuonProducer() override;

    static std::unique_ptr<PATMuonHeavyObjectCache> initializeGlobalCache(const edm::ParameterSet& iConfig) {
      return std::make_unique<PATMuonHeavyObjectCache>(iConfig);
    }

    static void globalEndJob(PATMuonHeavyObjectCache*) {}

    /// everything that needs to be done during the event loop
    void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;
    /// description of config file parameters
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    /// typedefs for convenience
    typedef edm::RefToBase<reco::Muon> MuonBaseRef;
    typedef std::vector<edm::Handle<edm::Association<reco::GenParticleCollection>>> GenAssociations;
    typedef std::vector<edm::Handle<edm::ValueMap<IsoDeposit>>> IsoDepositMaps;
    typedef std::vector<edm::Handle<edm::ValueMap<double>>> IsolationValueMaps;
    typedef std::pair<pat::IsolationKeys, edm::InputTag> IsolationLabel;
    typedef std::vector<IsolationLabel> IsolationLabels;

    /// common muon filling, for both the standard and PF2PAT case
    void fillMuon(Muon& aMuon,
                  const MuonBaseRef& muonRef,
                  const reco::CandidateBaseRef& baseRef,
                  const GenAssociations& genMatches,
                  const IsoDepositMaps& deposits,
                  const IsolationValueMaps& isolationValues) const;
    /// fill label vector from the contents of the parameter set,
    /// for the embedding of isoDeposits or userIsolation values
    template <typename T>
    void readIsolationLabels(const edm::ParameterSet& iConfig,
                             const char* psetName,
                             IsolationLabels& labels,
                             std::vector<edm::EDGetTokenT<edm::ValueMap<T>>>& tokens);

    void setMuonMiniIso(pat::Muon& aMuon, const pat::PackedCandidateCollection* pc);
    double getRelMiniIsoPUCorrected(const pat::Muon& muon, double rho, const std::vector<double>& area);

    double puppiCombinedIsolation(const pat::Muon& muon, const pat::PackedCandidateCollection* pc);
    bool isNeutralHadron(long pdgid);
    bool isChargedHadron(long pdgid);
    bool isPhoton(long pdgid);

    // embed various impact parameters with errors
    // embed high level selection
    void embedHighLevel(pat::Muon& aMuon,
                        reco::TrackRef track,
                        reco::TransientTrack& tt,
                        reco::Vertex& primaryVertex,
                        bool primaryVertexIsValid,
                        reco::BeamSpot& beamspot,
                        bool beamspotIsValid);
    double relMiniIsoPUCorrected(const pat::Muon& aMuon, double rho);
    std::optional<GlobalPoint> getMuonDirection(const reco::MuonChamberMatch& chamberMatch,
                                                const edm::ESHandle<GlobalTrackingGeometry>& geometry,
                                                const DetId& chamberId);
    void fillL1TriggerInfo(pat::Muon& muon,
                           edm::Handle<std::vector<pat::TriggerObjectStandAlone>>& triggerObjects,
                           const edm::TriggerNames& names,
                           const edm::ESHandle<GlobalTrackingGeometry>& geometry);
    void fillHltTriggerInfo(pat::Muon& muon,
                            edm::Handle<std::vector<pat::TriggerObjectStandAlone>>& triggerObjects,
                            const edm::TriggerNames& names,
                            const std::vector<std::string>& collection_names);

  private:
    /// input source
    edm::EDGetTokenT<edm::View<reco::Muon>> muonToken_;

    // for mini-iso calculation
    edm::EDGetTokenT<pat::PackedCandidateCollection> pcToken_;
    bool computeMiniIso_;
    bool computePuppiCombinedIso_;
    std::vector<double> effectiveAreaVec_;
    std::vector<double> miniIsoParams_;
    double relMiniIsoPUCorrected_;

    /// embed the track from best muon measurement (global pflow)
    bool embedBestTrack_;
    /// embed the track from best muon measurement (muon only)
    bool embedTunePBestTrack_;
    /// force separate embed of the best track even if already embedded
    bool forceEmbedBestTrack_;
    /// embed the track from inner tracker into the muon
    bool embedTrack_;
    /// embed track from muon system into the muon
    bool embedStandAloneMuon_;
    /// embed track of the combined fit into the muon
    bool embedCombinedMuon_;
    /// embed muon MET correction info for caloMET into the muon
    bool embedCaloMETMuonCorrs_;
    /// source of caloMET muon corrections
    edm::EDGetTokenT<edm::ValueMap<reco::MuonMETCorrectionData>> caloMETMuonCorrsToken_;
    /// embed muon MET correction info for tcMET into the muon
    bool embedTcMETMuonCorrs_;
    /// source of tcMET muon corrections
    edm::EDGetTokenT<edm::ValueMap<reco::MuonMETCorrectionData>> tcMETMuonCorrsToken_;
    /// embed track from picky muon fit into the muon
    bool embedPickyMuon_;
    /// embed track from tpfms muon fit into the muon
    bool embedTpfmsMuon_;
    /// embed track from DYT muon fit into the muon
    bool embedDytMuon_;
    /// add combined inverse beta measurement into the muon
    bool addInverseBeta_;
    /// input tag for reading inverse beta
    edm::EDGetTokenT<edm::ValueMap<reco::MuonTimeExtra>> muonTimeExtraToken_;
    /// add generator match information
    bool addGenMatch_;
    /// input tags for generator match information
    std::vector<edm::EDGetTokenT<edm::Association<reco::GenParticleCollection>>> genMatchTokens_;
    /// embed the gen match information into the muon
    bool embedGenMatch_;
    /// add resolutions to the muon (this will be data members of th muon even w/o embedding)
    bool addResolutions_;
    /// helper class to add resolutions to the muon
    pat::helper::KinResolutionsLoader resolutionLoader_;
    /// switch to use particle flow (PF2PAT) or not
    bool useParticleFlow_;
    /// input source pfCandidates that will be to be transformed into pat::Muons, when using PF2PAT
    edm::EDGetTokenT<reco::PFCandidateCollection> pfMuonToken_;
    /// embed pfCandidates into the muon
    bool embedPFCandidate_;
    /// embed high level selection variables
    bool embedHighLevelSelection_;
    /// input source of the primary vertex/beamspot
    edm::EDGetTokenT<reco::BeamSpot> beamLineToken_;
    /// input source of the primary vertex
    edm::EDGetTokenT<std::vector<reco::Vertex>> pvToken_;
    /// input source for isoDeposits
    IsolationLabels isoDepositLabels_;
    std::vector<edm::EDGetTokenT<edm::ValueMap<IsoDeposit>>> isoDepositTokens_;
    /// input source isolation value maps
    IsolationLabels isolationValueLabels_;
    std::vector<edm::EDGetTokenT<edm::ValueMap<double>>> isolationValueTokens_;
    /// add efficiencies to the muon (this will be data members of th muon even w/o embedding)
    bool addEfficiencies_;
    /// add user data to the muon (this will be data members of th muon even w/o embedding)
    bool useUserData_;
    /// add ecal PF energy
    bool embedPfEcalEnergy_;
    /// add puppi isolation
    bool addPuppiIsolation_;
    //PUPPI isolation tokens
    edm::EDGetTokenT<edm::ValueMap<float>> PUPPIIsolation_charged_hadrons_;
    edm::EDGetTokenT<edm::ValueMap<float>> PUPPIIsolation_neutral_hadrons_;
    edm::EDGetTokenT<edm::ValueMap<float>> PUPPIIsolation_photons_;
    //PUPPINoLeptons isolation tokens
    edm::EDGetTokenT<edm::ValueMap<float>> PUPPINoLeptonsIsolation_charged_hadrons_;
    edm::EDGetTokenT<edm::ValueMap<float>> PUPPINoLeptonsIsolation_neutral_hadrons_;
    edm::EDGetTokenT<edm::ValueMap<float>> PUPPINoLeptonsIsolation_photons_;
    /// standard muon selectors
    bool computeMuonIDMVA_;
    bool computeSoftMuonMVA_;
    bool recomputeBasicSelectors_;
    bool useJec_;
    edm::EDGetTokenT<reco::JetTagCollection> mvaBTagCollectionTag_;
    edm::EDGetTokenT<reco::JetCorrector> mvaL1Corrector_;
    edm::EDGetTokenT<reco::JetCorrector> mvaL1L2L3ResCorrector_;
    edm::EDGetTokenT<double> rho_;

    /// --- tools ---
    /// helper class to add userdefined isolation values to the muon
    pat::helper::MultiIsolator isolator_;
    /// isolation value pair for temporary storage before being folded into the muon
    pat::helper::MultiIsolator::IsolationValuePairs isolatorTmpStorage_;
    /// helper class to add efficiencies to the muon
    pat::helper::EfficiencyLoader efficiencyLoader_;
    /// helper class to add userData to the muon
    pat::PATUserDataHelper<pat::Muon> userDataHelper_;

    /// MC info
    edm::EDGetTokenT<edm::ValueMap<reco::MuonSimInfo>> simInfo_;

    /// Trigger
    bool addTriggerMatching_;
    edm::EDGetTokenT<std::vector<pat::TriggerObjectStandAlone>> triggerObjects_;
    edm::EDGetTokenT<edm::TriggerResults> triggerResults_;
    std::vector<std::string> hltCollectionFilters_;

    const edm::ESGetToken<GlobalTrackingGeometry, GlobalTrackingGeometryRecord> geometryToken_;
    const edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> transientTrackBuilderToken_;

    const edm::EDPutTokenT<std::vector<Muon>> patMuonPutToken_;
  };

}  // namespace pat

template <typename T>
void pat::PATMuonProducer::readIsolationLabels(const edm::ParameterSet& iConfig,
                                               const char* psetName,
                                               pat::PATMuonProducer::IsolationLabels& labels,
                                               std::vector<edm::EDGetTokenT<edm::ValueMap<T>>>& tokens) {
  labels.clear();

  if (iConfig.exists(psetName)) {
    edm::ParameterSet depconf = iConfig.getParameter<edm::ParameterSet>(psetName);

    if (depconf.exists("tracker"))
      labels.emplace_back(pat::TrackIso, depconf.getParameter<edm::InputTag>("tracker"));
    if (depconf.exists("ecal"))
      labels.emplace_back(pat::EcalIso, depconf.getParameter<edm::InputTag>("ecal"));
    if (depconf.exists("hcal"))
      labels.emplace_back(pat::HcalIso, depconf.getParameter<edm::InputTag>("hcal"));
    if (depconf.exists("pfAllParticles")) {
      labels.emplace_back(pat::PfAllParticleIso, depconf.getParameter<edm::InputTag>("pfAllParticles"));
    }
    if (depconf.exists("pfChargedHadrons")) {
      labels.emplace_back(pat::PfChargedHadronIso, depconf.getParameter<edm::InputTag>("pfChargedHadrons"));
    }
    if (depconf.exists("pfChargedAll")) {
      labels.emplace_back(pat::PfChargedAllIso, depconf.getParameter<edm::InputTag>("pfChargedAll"));
    }
    if (depconf.exists("pfPUChargedHadrons")) {
      labels.emplace_back(pat::PfPUChargedHadronIso, depconf.getParameter<edm::InputTag>("pfPUChargedHadrons"));
    }
    if (depconf.exists("pfNeutralHadrons")) {
      labels.emplace_back(pat::PfNeutralHadronIso, depconf.getParameter<edm::InputTag>("pfNeutralHadrons"));
    }
    if (depconf.exists("pfPhotons")) {
      labels.emplace_back(pat::PfGammaIso, depconf.getParameter<edm::InputTag>("pfPhotons"));
    }
    if (depconf.exists("user")) {
      std::vector<edm::InputTag> userdeps = depconf.getParameter<std::vector<edm::InputTag>>("user");
      std::vector<edm::InputTag>::const_iterator it = userdeps.begin(), ed = userdeps.end();
      int key = pat::IsolationKeys::UserBaseIso;
      for (; it != ed; ++it, ++key) {
        labels.push_back(std::make_pair(pat::IsolationKeys(key), *it));
      }
    }
  }
  tokens = edm::vector_transform(labels, [this](pat::PATMuonProducer::IsolationLabel const& label) {
    return consumes<edm::ValueMap<T>>(label.second);
  });
}

using namespace pat;
using namespace std;

PATMuonHeavyObjectCache::PATMuonHeavyObjectCache(const edm::ParameterSet& iConfig) {
  if (iConfig.getParameter<bool>("computeMiniIso")) {
    float mvaDrMax = iConfig.getParameter<double>("mvaDrMax");
    calculatePtRatioRel_ = std::make_unique<CalculatePtRatioRel>(mvaDrMax * mvaDrMax);
  }

  if (iConfig.getParameter<bool>("computeMuonIDMVA")) {
    edm::FileInPath mvaIDTrainingFile = iConfig.getParameter<edm::FileInPath>("mvaIDTrainingFile");
    muonMvaIDEstimator_ = std::make_unique<MuonMvaIDEstimator>(mvaIDTrainingFile);
  }

  if (iConfig.getParameter<bool>("computeSoftMuonMVA")) {
    edm::FileInPath softMvaTrainingFile = iConfig.getParameter<edm::FileInPath>("softMvaTrainingFile");
    softMuonMvaEstimator_ = std::make_unique<SoftMuonMvaEstimator>(softMvaTrainingFile);
  }
}

PATMuonProducer::PATMuonProducer(const edm::ParameterSet& iConfig, PATMuonHeavyObjectCache const*)
    : relMiniIsoPUCorrected_(0),
      useUserData_(iConfig.exists("userData")),
      computeMuonIDMVA_(false),
      computeSoftMuonMVA_(false),
      recomputeBasicSelectors_(false),
      useJec_(false),
      isolator_(iConfig.getParameter<edm::ParameterSet>("userIsolation"), consumesCollector(), false),
      geometryToken_{esConsumes()},
      transientTrackBuilderToken_{esConsumes(edm::ESInputTag("", "TransientTrackBuilder"))},
      patMuonPutToken_{produces<std::vector<Muon>>()} {
  // input source
  muonToken_ = consumes<edm::View<reco::Muon>>(iConfig.getParameter<edm::InputTag>("muonSource"));
  // embedding of tracks
  embedBestTrack_ = iConfig.getParameter<bool>("embedMuonBestTrack");
  embedTunePBestTrack_ = iConfig.getParameter<bool>("embedTunePMuonBestTrack");
  forceEmbedBestTrack_ = iConfig.getParameter<bool>("forceBestTrackEmbedding");
  embedTrack_ = iConfig.getParameter<bool>("embedTrack");
  embedCombinedMuon_ = iConfig.getParameter<bool>("embedCombinedMuon");
  embedStandAloneMuon_ = iConfig.getParameter<bool>("embedStandAloneMuon");
  // embedding of muon MET correction information
  embedCaloMETMuonCorrs_ = iConfig.getParameter<bool>("embedCaloMETMuonCorrs");
  embedTcMETMuonCorrs_ = iConfig.getParameter<bool>("embedTcMETMuonCorrs");
  caloMETMuonCorrsToken_ =
      mayConsume<edm::ValueMap<reco::MuonMETCorrectionData>>(iConfig.getParameter<edm::InputTag>("caloMETMuonCorrs"));
  tcMETMuonCorrsToken_ =
      mayConsume<edm::ValueMap<reco::MuonMETCorrectionData>>(iConfig.getParameter<edm::InputTag>("tcMETMuonCorrs"));
  // pflow specific configurables
  useParticleFlow_ = iConfig.getParameter<bool>("useParticleFlow");
  embedPFCandidate_ = iConfig.getParameter<bool>("embedPFCandidate");
  pfMuonToken_ = mayConsume<reco::PFCandidateCollection>(iConfig.getParameter<edm::InputTag>("pfMuonSource"));
  embedPfEcalEnergy_ = iConfig.getParameter<bool>("embedPfEcalEnergy");
  // embedding of tracks from TeV refit
  embedPickyMuon_ = iConfig.getParameter<bool>("embedPickyMuon");
  embedTpfmsMuon_ = iConfig.getParameter<bool>("embedTpfmsMuon");
  embedDytMuon_ = iConfig.getParameter<bool>("embedDytMuon");
  // embedding of inverse beta variable information
  addInverseBeta_ = iConfig.getParameter<bool>("addInverseBeta");
  if (addInverseBeta_) {
    muonTimeExtraToken_ =
        consumes<edm::ValueMap<reco::MuonTimeExtra>>(iConfig.getParameter<edm::InputTag>("sourceMuonTimeExtra"));
  }
  // Monte Carlo matching
  addGenMatch_ = iConfig.getParameter<bool>("addGenMatch");
  if (addGenMatch_) {
    embedGenMatch_ = iConfig.getParameter<bool>("embedGenMatch");
    if (iConfig.existsAs<edm::InputTag>("genParticleMatch")) {
      genMatchTokens_.push_back(consumes<edm::Association<reco::GenParticleCollection>>(
          iConfig.getParameter<edm::InputTag>("genParticleMatch")));
    } else {
      genMatchTokens_ = edm::vector_transform(
          iConfig.getParameter<std::vector<edm::InputTag>>("genParticleMatch"),
          [this](edm::InputTag const& tag) { return consumes<edm::Association<reco::GenParticleCollection>>(tag); });
    }
  }
  // efficiencies
  addEfficiencies_ = iConfig.getParameter<bool>("addEfficiencies");
  if (addEfficiencies_) {
    efficiencyLoader_ =
        pat::helper::EfficiencyLoader(iConfig.getParameter<edm::ParameterSet>("efficiencies"), consumesCollector());
  }
  // resolutions
  addResolutions_ = iConfig.getParameter<bool>("addResolutions");
  if (addResolutions_) {
    resolutionLoader_ =
        pat::helper::KinResolutionsLoader(iConfig.getParameter<edm::ParameterSet>("resolutions"), consumesCollector());
  }
  // puppi
  addPuppiIsolation_ = iConfig.getParameter<bool>("addPuppiIsolation");
  if (addPuppiIsolation_) {
    PUPPIIsolation_charged_hadrons_ =
        consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("puppiIsolationChargedHadrons"));
    PUPPIIsolation_neutral_hadrons_ =
        consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("puppiIsolationNeutralHadrons"));
    PUPPIIsolation_photons_ =
        consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("puppiIsolationPhotons"));
    //puppiNoLeptons
    PUPPINoLeptonsIsolation_charged_hadrons_ =
        consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("puppiNoLeptonsIsolationChargedHadrons"));
    PUPPINoLeptonsIsolation_neutral_hadrons_ =
        consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("puppiNoLeptonsIsolationNeutralHadrons"));
    PUPPINoLeptonsIsolation_photons_ =
        consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("puppiNoLeptonsIsolationPhotons"));
  }
  // read isoDeposit labels, for direct embedding
  readIsolationLabels(iConfig, "isoDeposits", isoDepositLabels_, isoDepositTokens_);
  // read isolation value labels, for direct embedding
  readIsolationLabels(iConfig, "isolationValues", isolationValueLabels_, isolationValueTokens_);
  // check to see if the user wants to add user data
  if (useUserData_) {
    userDataHelper_ = PATUserDataHelper<Muon>(iConfig.getParameter<edm::ParameterSet>("userData"), consumesCollector());
  }
  // embed high level selection variables
  embedHighLevelSelection_ = iConfig.getParameter<bool>("embedHighLevelSelection");
  if (embedHighLevelSelection_) {
    beamLineToken_ = consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamLineSrc"));
    pvToken_ = consumes<std::vector<reco::Vertex>>(iConfig.getParameter<edm::InputTag>("pvSrc"));
  }

  //for mini-isolation calculation
  computeMiniIso_ = iConfig.getParameter<bool>("computeMiniIso");

  computePuppiCombinedIso_ = iConfig.getParameter<bool>("computePuppiCombinedIso");

  effectiveAreaVec_ = iConfig.getParameter<std::vector<double>>("effectiveAreaVec");

  miniIsoParams_ = iConfig.getParameter<std::vector<double>>("miniIsoParams");
  if (computeMiniIso_ && miniIsoParams_.size() != 9) {
    throw cms::Exception("ParameterError") << "miniIsoParams must have exactly 9 elements.\n";
  }
  if (computeMiniIso_ || computePuppiCombinedIso_)
    pcToken_ = consumes<pat::PackedCandidateCollection>(iConfig.getParameter<edm::InputTag>("pfCandsForMiniIso"));

  // standard selectors
  recomputeBasicSelectors_ = iConfig.getParameter<bool>("recomputeBasicSelectors");
  computeMuonIDMVA_ = iConfig.getParameter<bool>("computeMuonIDMVA");

  if (computeMiniIso_) {
    // pfCombinedInclusiveSecondaryVertexV2BJetTags
    mvaBTagCollectionTag_ = consumes<reco::JetTagCollection>(iConfig.getParameter<edm::InputTag>("mvaJetTag"));
    mvaL1Corrector_ = consumes<reco::JetCorrector>(iConfig.getParameter<edm::InputTag>("mvaL1Corrector"));
    mvaL1L2L3ResCorrector_ = consumes<reco::JetCorrector>(iConfig.getParameter<edm::InputTag>("mvaL1L2L3ResCorrector"));
    rho_ = consumes<double>(iConfig.getParameter<edm::InputTag>("rho"));
    useJec_ = iConfig.getParameter<bool>("useJec");
  }

  computeSoftMuonMVA_ = iConfig.getParameter<bool>("computeSoftMuonMVA");

  // MC info
  simInfo_ = consumes<edm::ValueMap<reco::MuonSimInfo>>(iConfig.getParameter<edm::InputTag>("muonSimInfo"));

  addTriggerMatching_ = iConfig.getParameter<bool>("addTriggerMatching");
  if (addTriggerMatching_) {
    triggerObjects_ =
        consumes<std::vector<pat::TriggerObjectStandAlone>>(iConfig.getParameter<edm::InputTag>("triggerObjects"));
    triggerResults_ = consumes<edm::TriggerResults>(iConfig.getParameter<edm::InputTag>("triggerResults"));
  }
  hltCollectionFilters_ = iConfig.getParameter<std::vector<std::string>>("hltCollectionFilters");
}

PATMuonProducer::~PATMuonProducer() {}

std::optional<GlobalPoint> PATMuonProducer::getMuonDirection(const reco::MuonChamberMatch& chamberMatch,
                                                             const edm::ESHandle<GlobalTrackingGeometry>& geometry,
                                                             const DetId& chamberId) {
  const GeomDet* chamberGeometry = geometry->idToDet(chamberId);
  if (chamberGeometry) {
    LocalPoint localPosition(chamberMatch.x, chamberMatch.y, 0);
    return std::optional<GlobalPoint>(std::in_place, chamberGeometry->toGlobal(localPosition));
  }
  return std::optional<GlobalPoint>();
}

void PATMuonProducer::fillL1TriggerInfo(pat::Muon& aMuon,
                                        edm::Handle<std::vector<pat::TriggerObjectStandAlone>>& triggerObjects,
                                        const edm::TriggerNames& names,
                                        const edm::ESHandle<GlobalTrackingGeometry>& geometry) {
  // L1 trigger object parameters are defined at MB2/ME2. Use the muon
  // chamber matching information to get the local direction of the
  // muon trajectory and convert it to a global direction to match the
  // trigger objects

  std::optional<GlobalPoint> muonPosition;
  // Loop over chambers
  // initialize muonPosition with any available match, just in case
  // the second station is missing - it's better folling back to
  // dR matching at IP
  for (const auto& chamberMatch : aMuon.matches()) {
    if (chamberMatch.id.subdetId() == MuonSubdetId::DT) {
      DTChamberId detId(chamberMatch.id.rawId());
      if (abs(detId.station()) > 3)
        continue;
      muonPosition = getMuonDirection(chamberMatch, geometry, detId);
      if (abs(detId.station()) == 2)
        break;
    }
    if (chamberMatch.id.subdetId() == MuonSubdetId::CSC) {
      CSCDetId detId(chamberMatch.id.rawId());
      if (abs(detId.station()) > 3)
        continue;
      muonPosition = getMuonDirection(chamberMatch, geometry, detId);
      if (abs(detId.station()) == 2)
        break;
    }
  }
  if (not muonPosition)
    return;
  for (const auto& triggerObject : *triggerObjects) {
    if (triggerObject.hasTriggerObjectType(trigger::TriggerL1Mu)) {
      if (std::abs(triggerObject.eta()) < 0.001) {
        // L1 is defined in X-Y plain
        if (deltaPhi(triggerObject.phi(), muonPosition->phi()) > 0.1)
          continue;
      } else {
        // 3D L1
        if (deltaR(triggerObject.p4(), *muonPosition) > 0.15)
          continue;
      }
      pat::TriggerObjectStandAlone obj(triggerObject);
      obj.unpackPathNames(names);
      aMuon.addTriggerObjectMatch(obj);
    }
  }
}

void PATMuonProducer::fillHltTriggerInfo(pat::Muon& muon,
                                         edm::Handle<std::vector<pat::TriggerObjectStandAlone>>& triggerObjects,
                                         const edm::TriggerNames& names,
                                         const std::vector<std::string>& collection_filter_names) {
  // WARNING: in a case of close-by muons the dR matching may select both muons.
  // It's better to select the best match for a given collection.
  for (const auto& triggerObject : *triggerObjects) {
    if (triggerObject.hasTriggerObjectType(trigger::TriggerMuon)) {
      bool keepIt = false;
      for (const auto& name : collection_filter_names) {
        if (triggerObject.hasCollection(name)) {
          keepIt = true;
          break;
        }
      }
      if (not keepIt)
        continue;
      if (deltaR(triggerObject.p4(), muon) > 0.1)
        continue;
      pat::TriggerObjectStandAlone obj(triggerObject);
      obj.unpackPathNames(names);
      muon.addTriggerObjectMatch(obj);
    }
  }
}

void PATMuonProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // get the tracking Geometry
  auto geometry = iSetup.getHandle(geometryToken_);
  if (!geometry.isValid())
    throw cms::Exception("FatalError") << "Unable to find GlobalTrackingGeometryRecord in event!\n";

  // switch off embedding (in unschedules mode)
  if (iEvent.isRealData()) {
    addGenMatch_ = false;
    embedGenMatch_ = false;
  }

  edm::Handle<edm::View<reco::Muon>> muons;
  iEvent.getByToken(muonToken_, muons);

  edm::Handle<pat::PackedCandidateCollection> pc;
  if (computeMiniIso_ || computePuppiCombinedIso_)
    iEvent.getByToken(pcToken_, pc);

  // get the ESHandle for the transient track builder,
  // if needed for high level selection embedding
  TransientTrackBuilder const* trackBuilder = nullptr;

  if (isolator_.enabled())
    isolator_.beginEvent(iEvent, iSetup);
  if (efficiencyLoader_.enabled())
    efficiencyLoader_.newEvent(iEvent);
  if (resolutionLoader_.enabled())
    resolutionLoader_.newEvent(iEvent, iSetup);

  IsoDepositMaps deposits(isoDepositTokens_.size());
  for (size_t j = 0; j < isoDepositTokens_.size(); ++j) {
    iEvent.getByToken(isoDepositTokens_[j], deposits[j]);
  }

  IsolationValueMaps isolationValues(isolationValueTokens_.size());
  for (size_t j = 0; j < isolationValueTokens_.size(); ++j) {
    iEvent.getByToken(isolationValueTokens_[j], isolationValues[j]);
  }

  //value maps for puppi isolation
  edm::Handle<edm::ValueMap<float>> PUPPIIsolation_charged_hadrons;
  edm::Handle<edm::ValueMap<float>> PUPPIIsolation_neutral_hadrons;
  edm::Handle<edm::ValueMap<float>> PUPPIIsolation_photons;
  //value maps for puppiNoLeptons isolation
  edm::Handle<edm::ValueMap<float>> PUPPINoLeptonsIsolation_charged_hadrons;
  edm::Handle<edm::ValueMap<float>> PUPPINoLeptonsIsolation_neutral_hadrons;
  edm::Handle<edm::ValueMap<float>> PUPPINoLeptonsIsolation_photons;
  if (addPuppiIsolation_) {
    //puppi
    iEvent.getByToken(PUPPIIsolation_charged_hadrons_, PUPPIIsolation_charged_hadrons);
    iEvent.getByToken(PUPPIIsolation_neutral_hadrons_, PUPPIIsolation_neutral_hadrons);
    iEvent.getByToken(PUPPIIsolation_photons_, PUPPIIsolation_photons);
    //puppiNoLeptons
    iEvent.getByToken(PUPPINoLeptonsIsolation_charged_hadrons_, PUPPINoLeptonsIsolation_charged_hadrons);
    iEvent.getByToken(PUPPINoLeptonsIsolation_neutral_hadrons_, PUPPINoLeptonsIsolation_neutral_hadrons);
    iEvent.getByToken(PUPPINoLeptonsIsolation_photons_, PUPPINoLeptonsIsolation_photons);
  }

  // inputs for muon mva
  edm::Handle<reco::JetTagCollection> mvaBTagCollectionTag;
  edm::Handle<reco::JetCorrector> mvaL1Corrector;
  edm::Handle<reco::JetCorrector> mvaL1L2L3ResCorrector;
  if (computeMiniIso_) {
    iEvent.getByToken(mvaBTagCollectionTag_, mvaBTagCollectionTag);
    iEvent.getByToken(mvaL1Corrector_, mvaL1Corrector);
    iEvent.getByToken(mvaL1L2L3ResCorrector_, mvaL1L2L3ResCorrector);
  }

  // prepare the MC genMatchTokens_
  GenAssociations genMatches(genMatchTokens_.size());
  if (addGenMatch_) {
    for (size_t j = 0, nd = genMatchTokens_.size(); j < nd; ++j) {
      iEvent.getByToken(genMatchTokens_[j], genMatches[j]);
    }
  }

  // prepare the high level selection: needs beamline
  // OR primary vertex, depending on user selection
  reco::Vertex primaryVertex;
  reco::BeamSpot beamSpot;
  bool beamSpotIsValid = false;
  bool primaryVertexIsValid = false;
  if (embedHighLevelSelection_) {
    // get the beamspot
    edm::Handle<reco::BeamSpot> beamSpotHandle;
    iEvent.getByToken(beamLineToken_, beamSpotHandle);

    // get the primary vertex
    edm::Handle<std::vector<reco::Vertex>> pvHandle;
    iEvent.getByToken(pvToken_, pvHandle);

    if (beamSpotHandle.isValid()) {
      beamSpot = *beamSpotHandle;
      beamSpotIsValid = true;
    } else {
      edm::LogError("DataNotAvailable") << "No beam spot available from EventSetup, not adding high level selection \n";
    }
    if (pvHandle.isValid() && !pvHandle->empty()) {
      primaryVertex = pvHandle->at(0);
      primaryVertexIsValid = true;
    } else {
      edm::LogError("DataNotAvailable")
          << "No primary vertex available from EventSetup, not adding high level selection \n";
    }
    // this is needed by the IPTools methods from the tracking group
    trackBuilder = &iSetup.getData(transientTrackBuilderToken_);
  }

  // MC info
  edm::Handle<edm::ValueMap<reco::MuonSimInfo>> simInfo;
  bool simInfoIsAvailalbe = iEvent.getByToken(simInfo_, simInfo);

  // this will be the new object collection
  std::vector<Muon> patMuons;

  edm::Handle<reco::PFCandidateCollection> pfMuons;
  if (useParticleFlow_) {
    // get the PFCandidates of type muons
    iEvent.getByToken(pfMuonToken_, pfMuons);

    unsigned index = 0;
    for (reco::PFCandidateConstIterator i = pfMuons->begin(); i != pfMuons->end(); ++i, ++index) {
      const reco::PFCandidate& pfmu = *i;
      //const reco::IsolaPFCandidate& pfmu = *i;
      const reco::MuonRef& muonRef = pfmu.muonRef();
      assert(muonRef.isNonnull());

      MuonBaseRef muonBaseRef(muonRef);
      Muon aMuon(muonBaseRef);

      if (useUserData_) {
        userDataHelper_.add(aMuon, iEvent, iSetup);
      }

      // embed high level selection
      if (embedHighLevelSelection_) {
        // get the tracks
        reco::TrackRef innerTrack = muonBaseRef->innerTrack();
        reco::TrackRef globalTrack = muonBaseRef->globalTrack();
        reco::TrackRef bestTrack = muonBaseRef->muonBestTrack();
        reco::TrackRef chosenTrack = innerTrack;
        // Make sure the collection it points to is there
        if (bestTrack.isNonnull() && bestTrack.isAvailable())
          chosenTrack = bestTrack;

        if (chosenTrack.isNonnull() && chosenTrack.isAvailable()) {
          unsigned int nhits = chosenTrack->numberOfValidHits();  // ????
          aMuon.setNumberOfValidHits(nhits);

          reco::TransientTrack tt = trackBuilder->build(chosenTrack);
          embedHighLevel(aMuon, chosenTrack, tt, primaryVertex, primaryVertexIsValid, beamSpot, beamSpotIsValid);
        }

        if (globalTrack.isNonnull() && globalTrack.isAvailable() && !embedCombinedMuon_) {
          double norm_chi2 = globalTrack->chi2() / globalTrack->ndof();
          aMuon.setNormChi2(norm_chi2);
        }
      }
      reco::PFCandidateRef pfRef(pfMuons, index);
      //reco::PFCandidatePtr ptrToMother(pfMuons,index);
      reco::CandidateBaseRef pfBaseRef(pfRef);

      aMuon.setPFCandidateRef(pfRef);
      if (embedPFCandidate_)
        aMuon.embedPFCandidate();
      fillMuon(aMuon, muonBaseRef, pfBaseRef, genMatches, deposits, isolationValues);

      if (computeMiniIso_)
        setMuonMiniIso(aMuon, pc.product());

      if (addPuppiIsolation_) {
        aMuon.setIsolationPUPPI((*PUPPIIsolation_charged_hadrons)[muonBaseRef],
                                (*PUPPIIsolation_neutral_hadrons)[muonBaseRef],
                                (*PUPPIIsolation_photons)[muonBaseRef]);

        aMuon.setIsolationPUPPINoLeptons((*PUPPINoLeptonsIsolation_charged_hadrons)[muonBaseRef],
                                         (*PUPPINoLeptonsIsolation_neutral_hadrons)[muonBaseRef],
                                         (*PUPPINoLeptonsIsolation_photons)[muonBaseRef]);
      } else {
        aMuon.setIsolationPUPPI(-999., -999., -999.);
        aMuon.setIsolationPUPPINoLeptons(-999., -999., -999.);
      }

      if (embedPfEcalEnergy_) {
        aMuon.setPfEcalEnergy(pfmu.ecalEnergy());
      }

      patMuons.push_back(aMuon);
    }
  } else {
    edm::Handle<edm::View<reco::Muon>> muons;
    iEvent.getByToken(muonToken_, muons);

    // embedding of muon MET corrections
    edm::Handle<edm::ValueMap<reco::MuonMETCorrectionData>> caloMETMuonCorrs;
    //edm::ValueMap<reco::MuonMETCorrectionData> caloMETmuCorValueMap;
    if (embedCaloMETMuonCorrs_) {
      iEvent.getByToken(caloMETMuonCorrsToken_, caloMETMuonCorrs);
      //caloMETmuCorValueMap  = *caloMETmuCorValueMap_h;
    }
    edm::Handle<edm::ValueMap<reco::MuonMETCorrectionData>> tcMETMuonCorrs;
    //edm::ValueMap<reco::MuonMETCorrectionData> tcMETmuCorValueMap;
    if (embedTcMETMuonCorrs_) {
      iEvent.getByToken(tcMETMuonCorrsToken_, tcMETMuonCorrs);
      //tcMETmuCorValueMap  = *tcMETmuCorValueMap_h;
    }

    if (embedPfEcalEnergy_ || embedPFCandidate_) {
      // get the PFCandidates of type muons
      iEvent.getByToken(pfMuonToken_, pfMuons);
    }

    edm::Handle<edm::ValueMap<reco::MuonTimeExtra>> muonsTimeExtra;
    if (addInverseBeta_) {
      // get MuonTimerExtra value map
      iEvent.getByToken(muonTimeExtraToken_, muonsTimeExtra);
    }

    for (edm::View<reco::Muon>::const_iterator itMuon = muons->begin(); itMuon != muons->end(); ++itMuon) {
      // construct the Muon from the ref -> save ref to original object
      unsigned int idx = itMuon - muons->begin();
      MuonBaseRef muonRef = muons->refAt(idx);
      reco::CandidateBaseRef muonBaseRef(muonRef);

      Muon aMuon(muonRef);
      fillMuon(aMuon, muonRef, muonBaseRef, genMatches, deposits, isolationValues);
      if (computeMiniIso_)
        setMuonMiniIso(aMuon, pc.product());
      if (addPuppiIsolation_) {
        aMuon.setIsolationPUPPI((*PUPPIIsolation_charged_hadrons)[muonRef],
                                (*PUPPIIsolation_neutral_hadrons)[muonRef],
                                (*PUPPIIsolation_photons)[muonRef]);
        aMuon.setIsolationPUPPINoLeptons((*PUPPINoLeptonsIsolation_charged_hadrons)[muonRef],
                                         (*PUPPINoLeptonsIsolation_neutral_hadrons)[muonRef],
                                         (*PUPPINoLeptonsIsolation_photons)[muonRef]);
      } else {
        aMuon.setIsolationPUPPI(-999., -999., -999.);
        aMuon.setIsolationPUPPINoLeptons(-999., -999., -999.);
      }

      // Isolation
      if (isolator_.enabled()) {
        //reco::CandidatePtr mother =  ptrToMother->sourceCandidatePtr(0);
        isolator_.fill(*muons, idx, isolatorTmpStorage_);
        typedef pat::helper::MultiIsolator::IsolationValuePairs IsolationValuePairs;
        // better to loop backwards, so the vector is resized less times
        for (IsolationValuePairs::const_reverse_iterator it = isolatorTmpStorage_.rbegin(),
                                                         ed = isolatorTmpStorage_.rend();
             it != ed;
             ++it) {
          aMuon.setIsolation(it->first, it->second);
        }
      }

      //       for (size_t j = 0, nd = deposits.size(); j < nd; ++j) {
      // 	aMuon.setIsoDeposit(isoDepositLabels_[j].first,
      // 			    (*deposits[j])[muonRef]);
      //       }

      // add sel to selected
      edm::Ptr<reco::Muon> muonsPtr = muons->ptrAt(idx);
      if (useUserData_) {
        userDataHelper_.add(aMuon, iEvent, iSetup);
      }

      // embed high level selection
      if (embedHighLevelSelection_) {
        // get the tracks
        reco::TrackRef innerTrack = itMuon->innerTrack();
        reco::TrackRef globalTrack = itMuon->globalTrack();
        reco::TrackRef bestTrack = itMuon->muonBestTrack();
        reco::TrackRef chosenTrack = innerTrack;
        // Make sure the collection it points to is there
        if (bestTrack.isNonnull() && bestTrack.isAvailable())
          chosenTrack = bestTrack;
        if (chosenTrack.isNonnull() && chosenTrack.isAvailable()) {
          unsigned int nhits = chosenTrack->numberOfValidHits();  // ????
          aMuon.setNumberOfValidHits(nhits);

          reco::TransientTrack tt = trackBuilder->build(chosenTrack);
          embedHighLevel(aMuon, chosenTrack, tt, primaryVertex, primaryVertexIsValid, beamSpot, beamSpotIsValid);
        }

        if (globalTrack.isNonnull() && globalTrack.isAvailable()) {
          double norm_chi2 = globalTrack->chi2() / globalTrack->ndof();
          aMuon.setNormChi2(norm_chi2);
        }
      }

      // embed MET muon corrections
      if (embedCaloMETMuonCorrs_)
        aMuon.embedCaloMETMuonCorrs((*caloMETMuonCorrs)[muonRef]);
      if (embedTcMETMuonCorrs_)
        aMuon.embedTcMETMuonCorrs((*tcMETMuonCorrs)[muonRef]);

      if (embedPfEcalEnergy_ || embedPFCandidate_) {
        if (embedPfEcalEnergy_)
          aMuon.setPfEcalEnergy(-99.0);
        unsigned index = 0;
        for (const reco::PFCandidate& pfmu : *pfMuons) {
          if (pfmu.muonRef().isNonnull()) {
            if (pfmu.muonRef().id() != muonRef.id())
              throw cms::Exception("Configuration")
                  << "Muon reference within PF candidates does not point to the muon collection." << std::endl;
            if (pfmu.muonRef().key() == muonRef.key()) {
              reco::PFCandidateRef pfRef(pfMuons, index);
              aMuon.setPFCandidateRef(pfRef);
              if (embedPfEcalEnergy_)
                aMuon.setPfEcalEnergy(pfmu.ecalEnergy());
              if (embedPFCandidate_)
                aMuon.embedPFCandidate();
              break;
            }
          }
          index++;
        }
      }

      if (addInverseBeta_) {
        aMuon.readTimeExtra((*muonsTimeExtra)[muonRef]);
      }
      // MC info
      aMuon.initSimInfo();
      if (simInfoIsAvailalbe) {
        const auto& msi = (*simInfo)[muonBaseRef];
        aMuon.setSimType(msi.primaryClass);
        aMuon.setExtSimType(msi.extendedClass);
        aMuon.setSimFlavour(msi.flavour);
        aMuon.setSimHeaviestMotherFlavour(msi.heaviestMotherFlavour);
        aMuon.setSimPdgId(msi.pdgId);
        aMuon.setSimMotherPdgId(msi.motherPdgId);
        aMuon.setSimBX(msi.tpBX);
        aMuon.setSimTpEvent(msi.tpEvent);
        aMuon.setSimProdRho(msi.vertex.Rho());
        aMuon.setSimProdZ(msi.vertex.Z());
        aMuon.setSimPt(msi.p4.pt());
        aMuon.setSimEta(msi.p4.eta());
        aMuon.setSimPhi(msi.p4.phi());
        aMuon.setSimMatchQuality(msi.tpAssoQuality);
      }
      patMuons.push_back(aMuon);
    }
  }

  // sort muons in pt
  std::sort(patMuons.begin(), patMuons.end(), [](auto const& t1, auto const& t2) { return t1.pt() > t2.pt(); });

  // Store standard muon selection decisions and jet related
  // quantaties.
  // Need a separate loop over muons to have all inputs properly
  // computed and stored in the object.
  edm::Handle<double> rho;
  if (computeMiniIso_)
    iEvent.getByToken(rho_, rho);
  const reco::Vertex* pv(nullptr);
  if (primaryVertexIsValid)
    pv = &primaryVertex;

  edm::Handle<std::vector<pat::TriggerObjectStandAlone>> triggerObjects;
  edm::Handle<edm::TriggerResults> triggerResults;
  bool triggerObjectsAvailable = false;
  bool triggerResultsAvailable = false;
  if (addTriggerMatching_) {
    triggerObjectsAvailable = iEvent.getByToken(triggerObjects_, triggerObjects);
    triggerResultsAvailable = iEvent.getByToken(triggerResults_, triggerResults);
  }

  for (auto& muon : patMuons) {
    // trigger info
    if (addTriggerMatching_ and triggerObjectsAvailable and triggerResultsAvailable) {
      const edm::TriggerNames& triggerNames(iEvent.triggerNames(*triggerResults));
      fillL1TriggerInfo(muon, triggerObjects, triggerNames, geometry);
      fillHltTriggerInfo(muon, triggerObjects, triggerNames, hltCollectionFilters_);
    }

    if (recomputeBasicSelectors_) {
      muon.setSelectors(0);
      bool isRun2016BCDEF = (272728 <= iEvent.run() && iEvent.run() <= 278808);
      muon.setSelectors(muon::makeSelectorBitset(muon, pv, isRun2016BCDEF));
    }
    float miniIsoValue = -1;
    if (computeMiniIso_) {
      // MiniIsolation working points

      miniIsoValue = getRelMiniIsoPUCorrected(muon, *rho, effectiveAreaVec_);

      muon.setSelector(reco::Muon::MiniIsoLoose, miniIsoValue < 0.40);
      muon.setSelector(reco::Muon::MiniIsoMedium, miniIsoValue < 0.20);
      muon.setSelector(reco::Muon::MiniIsoTight, miniIsoValue < 0.10);
      muon.setSelector(reco::Muon::MiniIsoVeryTight, miniIsoValue < 0.05);
    }

    double puppiCombinedIsolationPAT = -1;
    if (computePuppiCombinedIso_) {
      puppiCombinedIsolationPAT = puppiCombinedIsolation(muon, pc.product());
      muon.setSelector(reco::Muon::PuppiIsoLoose, puppiCombinedIsolationPAT < 0.27);
      muon.setSelector(reco::Muon::PuppiIsoMedium, puppiCombinedIsolationPAT < 0.22);
      muon.setSelector(reco::Muon::PuppiIsoTight, puppiCombinedIsolationPAT < 0.12);
    }

    std::array<float, 2> jetPtRatioRel = {{0.0, 0.0}};
    if (primaryVertexIsValid && computeMiniIso_) {
      if (useJec_) {
        jetPtRatioRel = globalCache()->calculatePtRatioRel().computePtRatioRel(
            muon, *(mvaBTagCollectionTag.product()), mvaL1Corrector.product(), mvaL1L2L3ResCorrector.product());
      } else {
        jetPtRatioRel = globalCache()->calculatePtRatioRel().computePtRatioRel(muon, *mvaBTagCollectionTag);
      }

      muon.setJetPtRatio(jetPtRatioRel[0]);
      muon.setJetPtRel(jetPtRatioRel[1]);

      // multi-isolation
      if (computeMiniIso_) {
        muon.setSelector(reco::Muon::MultiIsoMedium,
                         miniIsoValue < 0.11 && (muon.jetPtRatio() > 0.74 || muon.jetPtRel() > 6.8));
      }
    }

    // MVA ID
    float mvaID = 0.0;
    constexpr int MVAsentinelValue = -99;
    constexpr float mvaIDmediumCut = 0.08;
    constexpr float mvaIDtightCut = 0.12;
    if (computeMuonIDMVA_) {
      const double dz = std::abs(muon.muonBestTrack()->dz(primaryVertex.position()));
      const double dxy = std::abs(muon.muonBestTrack()->dxy(primaryVertex.position()));
      if (muon.isLooseMuon()) {
        mvaID = globalCache()->muonMvaIDEstimator().computeMVAID(muon)[1];
      } else {
        mvaID = MVAsentinelValue;
      }
      muon.setMvaIDValue(mvaID);
      muon.setSelector(reco::Muon::MvaIDwpMedium, muon.mvaIDValue() > mvaIDmediumCut);
      muon.setSelector(reco::Muon::MvaIDwpTight, muon.mvaIDValue() > mvaIDtightCut and dz < 0.5 and dxy < 0.2);
    }

    //SOFT MVA
    if (computeSoftMuonMVA_) {
      float mva = globalCache()->softMuonMvaEstimator().computeMva(muon);
      muon.setSoftMvaValue(mva);
      //preselection in SoftMuonMvaEstimator.cc
      muon.setSelector(reco::Muon::SoftMvaId, muon.softMvaValue() > 0.58);  //WP choose for bmm4
    }
  }

  // put products in Event
  iEvent.emplace(patMuonPutToken_, std::move(patMuons));

  if (isolator_.enabled())
    isolator_.endEvent();
}

void PATMuonProducer::fillMuon(Muon& aMuon,
                               const MuonBaseRef& muonRef,
                               const reco::CandidateBaseRef& baseRef,
                               const GenAssociations& genMatches,
                               const IsoDepositMaps& deposits,
                               const IsolationValueMaps& isolationValues) const {
  // in the particle flow algorithm,
  // the muon momentum is recomputed.
  // the new value is stored as the momentum of the
  // resulting PFCandidate of type Muon, and choosen
  // as the pat::Muon momentum
  if (useParticleFlow_)
    aMuon.setP4(aMuon.pfCandidateRef()->p4());
  if (embedTrack_)
    aMuon.embedTrack();
  if (embedStandAloneMuon_)
    aMuon.embedStandAloneMuon();
  if (embedCombinedMuon_)
    aMuon.embedCombinedMuon();

  // embed the TeV refit track refs (only available for globalMuons)
  if (aMuon.isGlobalMuon()) {
    if (embedPickyMuon_ && aMuon.isAValidMuonTrack(reco::Muon::Picky))
      aMuon.embedPickyMuon();
    if (embedTpfmsMuon_ && aMuon.isAValidMuonTrack(reco::Muon::TPFMS))
      aMuon.embedTpfmsMuon();
    if (embedDytMuon_ && aMuon.isAValidMuonTrack(reco::Muon::DYT))
      aMuon.embedDytMuon();
  }

  // embed best tracks (at the end, so unless forceEmbedBestTrack_ is true we can save some space not embedding them twice)
  if (embedBestTrack_)
    aMuon.embedMuonBestTrack(forceEmbedBestTrack_);
  if (embedTunePBestTrack_)
    aMuon.embedTunePMuonBestTrack(forceEmbedBestTrack_);

  // store the match to the generated final state muons
  if (addGenMatch_) {
    for (auto const& genMatch : genMatches) {
      reco::GenParticleRef genMuon = (*genMatch)[baseRef];
      aMuon.addGenParticleRef(genMuon);
    }
    if (embedGenMatch_)
      aMuon.embedGenParticle();
  }
  if (efficiencyLoader_.enabled()) {
    efficiencyLoader_.setEfficiencies(aMuon, muonRef);
  }

  for (size_t j = 0, nd = deposits.size(); j < nd; ++j) {
    if (useParticleFlow_) {
      if (deposits[j]->contains(baseRef.id())) {
        aMuon.setIsoDeposit(isoDepositLabels_[j].first, (*deposits[j])[baseRef]);
      } else if (deposits[j]->contains(muonRef.id())) {
        aMuon.setIsoDeposit(isoDepositLabels_[j].first, (*deposits[j])[muonRef]);
      } else {
        reco::CandidatePtr source = aMuon.pfCandidateRef()->sourceCandidatePtr(0);
        aMuon.setIsoDeposit(isoDepositLabels_[j].first, (*deposits[j])[source]);
      }
    } else {
      aMuon.setIsoDeposit(isoDepositLabels_[j].first, (*deposits[j])[muonRef]);
    }
  }

  for (size_t j = 0; j < isolationValues.size(); ++j) {
    if (useParticleFlow_) {
      if (isolationValues[j]->contains(baseRef.id())) {
        aMuon.setIsolation(isolationValueLabels_[j].first, (*isolationValues[j])[baseRef]);
      } else if (isolationValues[j]->contains(muonRef.id())) {
        aMuon.setIsolation(isolationValueLabels_[j].first, (*isolationValues[j])[muonRef]);
      } else {
        reco::CandidatePtr source = aMuon.pfCandidateRef()->sourceCandidatePtr(0);
        aMuon.setIsolation(isolationValueLabels_[j].first, (*isolationValues[j])[source]);
      }
    } else {
      aMuon.setIsolation(isolationValueLabels_[j].first, (*isolationValues[j])[muonRef]);
    }
  }

  if (resolutionLoader_.enabled()) {
    resolutionLoader_.setResolutions(aMuon);
  }
}

void PATMuonProducer::setMuonMiniIso(Muon& aMuon, const PackedCandidateCollection* pc) {
  pat::PFIsolation miniiso = pat::getMiniPFIsolation(pc,
                                                     aMuon.polarP4(),
                                                     miniIsoParams_[0],
                                                     miniIsoParams_[1],
                                                     miniIsoParams_[2],
                                                     miniIsoParams_[3],
                                                     miniIsoParams_[4],
                                                     miniIsoParams_[5],
                                                     miniIsoParams_[6],
                                                     miniIsoParams_[7],
                                                     miniIsoParams_[8]);
  aMuon.setMiniPFIsolation(miniiso);
}

double PATMuonProducer::getRelMiniIsoPUCorrected(const pat::Muon& muon, double rho, const std::vector<double>& area) {
  double mindr(miniIsoParams_[0]);
  double maxdr(miniIsoParams_[1]);
  double kt_scale(miniIsoParams_[2]);
  double drcut = pat::miniIsoDr(muon.polarP4(), mindr, maxdr, kt_scale);
  return pat::muonRelMiniIsoPUCorrected(muon.miniPFIsolation(), muon.polarP4(), drcut, rho, area);
}

double PATMuonProducer::puppiCombinedIsolation(const pat::Muon& muon, const pat::PackedCandidateCollection* pc) {
  constexpr double dR_threshold = 0.4;
  constexpr double dR2_threshold = dR_threshold * dR_threshold;
  constexpr double mix_fraction = 0.5;
  enum particleType { CH = 0, NH = 1, PH = 2, OTHER = 100000 };
  double val_PuppiWithLep = 0.0;
  double val_PuppiWithoutLep = 0.0;

  for (const auto& cand : *pc) {  //pat::pat::PackedCandidate loop start

    const particleType pType = isChargedHadron(cand.pdgId())   ? CH
                               : isNeutralHadron(cand.pdgId()) ? NH
                               : isPhoton(cand.pdgId())        ? PH
                                                               : OTHER;
    if (pType == OTHER) {
      if (cand.pdgId() != 1 && cand.pdgId() != 2 && abs(cand.pdgId()) != 11 && abs(cand.pdgId()) != 13) {
        LogTrace("PATMuonProducer") << "candidate with PDGID = " << cand.pdgId()
                                    << " is not CH/NH/PH/e/mu or 1/2 (and this is removed from isolation calculation)"
                                    << std::endl;
      }
      continue;
    }
    double d_eta = std::abs(cand.eta() - muon.eta());
    if (d_eta > dR_threshold)
      continue;

    double d_phi = std::abs(reco::deltaPhi(cand.phi(), muon.phi()));
    if (d_phi > dR_threshold)
      continue;

    double dR2 = reco::deltaR2(cand, muon);
    if (dR2 > dR2_threshold)
      continue;
    if (pType == CH && dR2 < 0.0001 * 0.0001)
      continue;
    if (pType == NH && dR2 < 0.01 * 0.01)
      continue;
    if (pType == PH && dR2 < 0.01 * 0.01)
      continue;
    val_PuppiWithLep += cand.pt() * cand.puppiWeight();
    val_PuppiWithoutLep += cand.pt() * cand.puppiWeightNoLep();

  }  //pat::pat::PackedCandidate loop end

  double reliso_Puppi_withLep = val_PuppiWithLep / muon.pt();
  double reliso_Puppi_withoutlep = val_PuppiWithoutLep / muon.pt();
  double reliso_Puppi_combined = mix_fraction * reliso_Puppi_withLep + (1.0 - mix_fraction) * reliso_Puppi_withoutlep;
  return reliso_Puppi_combined;
}

bool PATMuonProducer::isNeutralHadron(long pdgid) { return std::abs(pdgid) == 130; }

bool PATMuonProducer::isChargedHadron(long pdgid) { return std::abs(pdgid) == 211; }

bool PATMuonProducer::isPhoton(long pdgid) { return pdgid == 22; }

// ParameterSet description for module
void PATMuonProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription iDesc;
  iDesc.setComment("PAT muon producer module");

  // input source
  iDesc.add<edm::InputTag>("muonSource", edm::InputTag("no default"))->setComment("input collection");

  // embedding
  iDesc.add<bool>("embedMuonBestTrack", true)->setComment("embed muon best track (global pflow)");
  iDesc.add<bool>("embedTunePMuonBestTrack", true)->setComment("embed muon best track (muon only)");
  iDesc.add<bool>("forceBestTrackEmbedding", true)
      ->setComment(
          "force embedding separately the best tracks even if they're already embedded e.g. as tracker or global "
          "tracks");
  iDesc.add<bool>("embedTrack", true)->setComment("embed external track");
  iDesc.add<bool>("embedStandAloneMuon", true)->setComment("embed external stand-alone muon");
  iDesc.add<bool>("embedCombinedMuon", false)->setComment("embed external combined muon");
  iDesc.add<bool>("embedPickyMuon", false)->setComment("embed external picky track");
  iDesc.add<bool>("embedTpfmsMuon", false)->setComment("embed external tpfms track");
  iDesc.add<bool>("embedDytMuon", false)->setComment("embed external dyt track ");

  // embedding of MET muon corrections
  iDesc.add<bool>("embedCaloMETMuonCorrs", true)->setComment("whether to add MET muon correction for caloMET or not");
  iDesc.add<edm::InputTag>("caloMETMuonCorrs", edm::InputTag("muonMETValueMapProducer", "muCorrData"))
      ->setComment("source of MET muon corrections for caloMET");
  iDesc.add<bool>("embedTcMETMuonCorrs", true)->setComment("whether to add MET muon correction for tcMET or not");
  iDesc.add<edm::InputTag>("tcMETMuonCorrs", edm::InputTag("muonTCMETValueMapProducer", "muCorrData"))
      ->setComment("source of MET muon corrections for tcMET");

  // pf specific parameters
  iDesc.add<edm::InputTag>("pfMuonSource", edm::InputTag("pfMuons"))->setComment("particle flow input collection");
  iDesc.add<bool>("useParticleFlow", false)->setComment("whether to use particle flow or not");
  iDesc.add<bool>("embedPFCandidate", false)->setComment("embed external particle flow object");
  iDesc.add<bool>("embedPfEcalEnergy", true)->setComment("add ecal energy as reconstructed by PF");

  // inverse beta computation
  iDesc.add<bool>("addInverseBeta", true)->setComment("add combined inverse beta");
  iDesc.add<edm::InputTag>("sourceInverseBeta", edm::InputTag("muons", "combined"))
      ->setComment("source of inverse beta values");

  // MC matching configurables
  iDesc.add<bool>("addGenMatch", true)->setComment("add MC matching");
  iDesc.add<bool>("embedGenMatch", false)->setComment("embed MC matched MC information");
  std::vector<edm::InputTag> emptySourceVector;
  iDesc
      .addNode(edm::ParameterDescription<edm::InputTag>("genParticleMatch", edm::InputTag(), true) xor
               edm::ParameterDescription<std::vector<edm::InputTag>>("genParticleMatch", emptySourceVector, true))
      ->setComment("input with MC match information");

  // mini-iso
  iDesc.add<bool>("computeMiniIso", false)->setComment("whether or not to compute and store electron mini-isolation");
  iDesc.add<bool>("computePuppiCombinedIso", false)
      ->setComment("whether or not to compute and store puppi combined isolation");

  iDesc.add<edm::InputTag>("pfCandsForMiniIso", edm::InputTag("packedPFCandidates"))
      ->setComment("collection to use to compute mini-iso");
  iDesc.add<std::vector<double>>("miniIsoParams", std::vector<double>())
      ->setComment("mini-iso parameters to use for muons");

  iDesc.add<bool>("addTriggerMatching", false)->setComment("add L1 and HLT matching to offline muon");

  pat::helper::KinResolutionsLoader::fillDescription(iDesc);

  // IsoDeposit configurables
  edm::ParameterSetDescription isoDepositsPSet;
  isoDepositsPSet.addOptional<edm::InputTag>("tracker");
  isoDepositsPSet.addOptional<edm::InputTag>("ecal");
  isoDepositsPSet.addOptional<edm::InputTag>("hcal");
  isoDepositsPSet.addOptional<edm::InputTag>("particle");
  isoDepositsPSet.addOptional<edm::InputTag>("pfChargedHadrons");
  isoDepositsPSet.addOptional<edm::InputTag>("pfChargedAll");
  isoDepositsPSet.addOptional<edm::InputTag>("pfPUChargedHadrons");
  isoDepositsPSet.addOptional<edm::InputTag>("pfNeutralHadrons");
  isoDepositsPSet.addOptional<edm::InputTag>("pfPhotons");
  isoDepositsPSet.addOptional<std::vector<edm::InputTag>>("user");
  iDesc.addOptional("isoDeposits", isoDepositsPSet);

  // isolation values configurables
  edm::ParameterSetDescription isolationValuesPSet;
  isolationValuesPSet.addOptional<edm::InputTag>("tracker");
  isolationValuesPSet.addOptional<edm::InputTag>("ecal");
  isolationValuesPSet.addOptional<edm::InputTag>("hcal");
  isolationValuesPSet.addOptional<edm::InputTag>("particle");
  isolationValuesPSet.addOptional<edm::InputTag>("pfChargedHadrons");
  isolationValuesPSet.addOptional<edm::InputTag>("pfChargedAll");
  isolationValuesPSet.addOptional<edm::InputTag>("pfPUChargedHadrons");
  isolationValuesPSet.addOptional<edm::InputTag>("pfNeutralHadrons");
  isolationValuesPSet.addOptional<edm::InputTag>("pfPhotons");
  iDesc.addOptional("isolationValues", isolationValuesPSet);

  iDesc.ifValue(edm::ParameterDescription<bool>("addPuppiIsolation", false, true),
                true >> (edm::ParameterDescription<edm::InputTag>(
                             "puppiIsolationChargedHadrons",
                             edm::InputTag("muonPUPPIIsolation", "h+-DR030-ThresholdVeto000-ConeVeto000"),
                             true) and
                         edm::ParameterDescription<edm::InputTag>(
                             "puppiIsolationNeutralHadrons",
                             edm::InputTag("muonPUPPIIsolation", "h0-DR030-ThresholdVeto000-ConeVeto001"),
                             true) and
                         edm::ParameterDescription<edm::InputTag>(
                             "puppiIsolationPhotons",
                             edm::InputTag("muonPUPPIIsolation", "gamma-DR030-ThresholdVeto000-ConeVeto001"),
                             true) and
                         edm::ParameterDescription<edm::InputTag>(
                             "puppiNoLeptonsIsolationChargedHadrons",
                             edm::InputTag("muonPUPPINoLeptonsIsolation", "h+-DR030-ThresholdVeto000-ConeVeto000"),
                             true) and
                         edm::ParameterDescription<edm::InputTag>(
                             "puppiNoLeptonsIsolationNeutralHadrons",
                             edm::InputTag("muonPUPPINoLeptonsIsolation", "h0-DR030-ThresholdVeto000-ConeVeto001"),
                             true) and
                         edm::ParameterDescription<edm::InputTag>(
                             "puppiNoLeptonsIsolationPhotons",
                             edm::InputTag("muonPUPPINoLeptonsIsolation", "gamma-DR030-ThresholdVeto000-ConeVeto001"),
                             true)) or
                    false >> edm::EmptyGroupDescription());

  // Efficiency configurables
  edm::ParameterSetDescription efficienciesPSet;
  efficienciesPSet.setAllowAnything();  // TODO: the pat helper needs to implement a description.
  iDesc.add("efficiencies", efficienciesPSet);
  iDesc.add<bool>("addEfficiencies", false);

  // Check to see if the user wants to add user data
  edm::ParameterSetDescription userDataPSet;
  PATUserDataHelper<Muon>::fillDescription(userDataPSet);
  iDesc.addOptional("userData", userDataPSet);

  edm::ParameterSetDescription isolationPSet;
  isolationPSet.setAllowAnything();  // TODO: the pat helper needs to implement a description.
  iDesc.add("userIsolation", isolationPSet);

  iDesc.add<bool>("embedHighLevelSelection", true)->setComment("embed high level selection");
  edm::ParameterSetDescription highLevelPSet;
  highLevelPSet.setAllowAnything();
  iDesc.addNode(edm::ParameterDescription<edm::InputTag>("beamLineSrc", edm::InputTag(), true))
      ->setComment("input with high level selection");
  iDesc.addNode(edm::ParameterDescription<edm::InputTag>("pvSrc", edm::InputTag(), true))
      ->setComment("input with high level selection");

  //descriptions.add("PATMuonProducer", iDesc);
}

// embed various impact parameters with errors
// embed high level selection
void PATMuonProducer::embedHighLevel(pat::Muon& aMuon,
                                     reco::TrackRef track,
                                     reco::TransientTrack& tt,
                                     reco::Vertex& primaryVertex,
                                     bool primaryVertexIsValid,
                                     reco::BeamSpot& beamspot,
                                     bool beamspotIsValid) {
  // Correct to PV

  // PV2D
  aMuon.setDB(track->dxy(primaryVertex.position()),
              track->dxyError(primaryVertex.position(), primaryVertex.covariance()),
              pat::Muon::PV2D);

  // PV3D
  std::pair<bool, Measurement1D> result =
      IPTools::signedImpactParameter3D(tt, GlobalVector(track->px(), track->py(), track->pz()), primaryVertex);
  double d0_corr = result.second.value();
  double d0_err = primaryVertexIsValid ? result.second.error() : -1.0;
  aMuon.setDB(d0_corr, d0_err, pat::Muon::PV3D);

  // Correct to beam spot

  // BS2D
  aMuon.setDB(track->dxy(beamspot), track->dxyError(beamspot), pat::Muon::BS2D);

  // make a fake vertex out of beam spot
  reco::Vertex vBeamspot(beamspot.position(), beamspot.rotatedCovariance3D());

  // BS3D
  result = IPTools::signedImpactParameter3D(tt, GlobalVector(track->px(), track->py(), track->pz()), vBeamspot);
  d0_corr = result.second.value();
  d0_err = beamspotIsValid ? result.second.error() : -1.0;
  aMuon.setDB(d0_corr, d0_err, pat::Muon::BS3D);

  // PVDZ
  aMuon.setDB(
      track->dz(primaryVertex.position()), std::hypot(track->dzError(), primaryVertex.zError()), pat::Muon::PVDZ);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PATMuonProducer);

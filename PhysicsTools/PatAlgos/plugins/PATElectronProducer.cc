/**
  \class    pat::PATElectronProducer PATElectronProducer.h "PhysicsTools/PatAlgos/interface/PATElectronProducer.h"
  \brief    Produces pat::Electron's

   The PATElectronProducer produces analysis-level pat::Electron's starting from
   a collection of objects of reco::GsfElectron.

  \author   Steven Lowette, James Lamb\
  \version  $Id: PATElectronProducer.h,v 1.31 2013/02/27 23:26:56 wmtan Exp $
*/

#include "CommonTools/Egamma/interface/ConversionTools.h"
#include "CommonTools/Utils/interface/PtComparator.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Candidate/interface/CandAssociation.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/PFIsolation.h"
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
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "FWCore/Utilities/interface/transform.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/CaloTopologyRecord.h"
#include "PhysicsTools/PatAlgos/interface/EfficiencyLoader.h"
#include "PhysicsTools/PatAlgos/interface/KinResolutionsLoader.h"
#include "PhysicsTools/PatAlgos/interface/MultiIsolator.h"
#include "PhysicsTools/PatAlgos/interface/PATUserDataHelper.h"
#include "PhysicsTools/PatUtils/interface/CaloIsolationEnergy.h"
#include "PhysicsTools/PatUtils/interface/MiniIsolation.h"
#include "PhysicsTools/PatUtils/interface/TrackerIsolationPt.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"

#include <memory>
#include <string>
#include <vector>

namespace pat {

  class TrackerIsolationPt;
  class CaloIsolationEnergy;
  class LeptonLRCalc;

  class PATElectronProducer : public edm::stream::EDProducer<> {
  public:
    explicit PATElectronProducer(const edm::ParameterSet& iConfig);
    ~PATElectronProducer() override;

    void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    // configurables
    const edm::EDGetTokenT<edm::View<reco::GsfElectron>> electronToken_;
    const edm::EDGetTokenT<reco::ConversionCollection> hConversionsToken_;
    const bool embedGsfElectronCore_;
    const bool embedGsfTrack_;
    const bool embedSuperCluster_;
    const bool embedPflowSuperCluster_;
    const bool embedSeedCluster_;
    const bool embedBasicClusters_;
    const bool embedPreshowerClusters_;
    const bool embedPflowBasicClusters_;
    const bool embedPflowPreshowerClusters_;
    const bool embedTrack_;
    bool addGenMatch_;
    bool embedGenMatch_;
    const bool embedRecHits_;
    // for mini-iso calculation
    edm::EDGetTokenT<pat::PackedCandidateCollection> pcToken_;
    bool computeMiniIso_;
    std::vector<double> miniIsoParamsE_;
    std::vector<double> miniIsoParamsB_;

    typedef std::vector<edm::Handle<edm::Association<reco::GenParticleCollection>>> GenAssociations;

    std::vector<edm::EDGetTokenT<edm::Association<reco::GenParticleCollection>>> genMatchTokens_;

    /// pflow specific
    const bool useParticleFlow_;
    const bool usePfCandidateMultiMap_;
    const edm::EDGetTokenT<reco::PFCandidateCollection> pfElecToken_;
    const edm::EDGetTokenT<edm::ValueMap<reco::PFCandidatePtr>> pfCandidateMapToken_;
    const edm::EDGetTokenT<edm::ValueMap<std::vector<reco::PFCandidateRef>>> pfCandidateMultiMapToken_;
    const bool embedPFCandidate_;

    /// mva input variables
    const bool addMVAVariables_;
    const edm::InputTag reducedBarrelRecHitCollection_;
    const edm::EDGetTokenT<EcalRecHitCollection> reducedBarrelRecHitCollectionToken_;
    const edm::InputTag reducedEndcapRecHitCollection_;
    const edm::EDGetTokenT<EcalRecHitCollection> reducedEndcapRecHitCollectionToken_;
    const EcalClusterLazyTools::ESGetTokens ecalClusterToolsESGetTokens_;

    const bool addPFClusterIso_;
    const bool addPuppiIsolation_;
    const edm::EDGetTokenT<edm::ValueMap<float>> ecalPFClusterIsoT_;
    const edm::EDGetTokenT<edm::ValueMap<float>> hcalPFClusterIsoT_;

    /// embed high level selection variables?
    const bool embedHighLevelSelection_;
    const edm::EDGetTokenT<reco::BeamSpot> beamLineToken_;
    const edm::EDGetTokenT<std::vector<reco::Vertex>> pvToken_;

    typedef edm::RefToBase<reco::GsfElectron> ElectronBaseRef;
    typedef std::vector<edm::Handle<edm::ValueMap<IsoDeposit>>> IsoDepositMaps;
    typedef std::vector<edm::Handle<edm::ValueMap<double>>> IsolationValueMaps;

    /// common electron filling, for both the standard and PF2PAT case
    void fillElectron(Electron& aElectron,
                      const ElectronBaseRef& electronRef,
                      const reco::CandidateBaseRef& baseRef,
                      const GenAssociations& genMatches,
                      const IsoDepositMaps& deposits,
                      const bool pfId,
                      const IsolationValueMaps& isolationValues,
                      const IsolationValueMaps& isolationValuesNoPFId) const;

    void fillElectron2(Electron& anElectron,
                       const reco::CandidatePtr& candPtrForIsolation,
                       const reco::CandidatePtr& candPtrForGenMatch,
                       const reco::CandidatePtr& candPtrForLoader,
                       const GenAssociations& genMatches,
                       const IsoDepositMaps& deposits,
                       const IsolationValueMaps& isolationValues) const;

    // set the mini-isolation variables
    void setElectronMiniIso(pat::Electron& anElectron, const pat::PackedCandidateCollection* pc);

    // embed various impact parameters with errors
    // embed high level selection
    void embedHighLevel(pat::Electron& anElectron,
                        reco::GsfTrackRef track,
                        reco::TransientTrack& tt,
                        reco::Vertex& primaryVertex,
                        bool primaryVertexIsValid,
                        reco::BeamSpot& beamspot,
                        bool beamspotIsValid);

    typedef std::pair<pat::IsolationKeys, edm::InputTag> IsolationLabel;
    typedef std::vector<IsolationLabel> IsolationLabels;

    /// fill the labels vector from the contents of the parameter set,
    /// for the isodeposit or isolation values embedding
    template <typename T>
    void readIsolationLabels(const edm::ParameterSet& iConfig,
                             const char* psetName,
                             IsolationLabels& labels,
                             std::vector<edm::EDGetTokenT<edm::ValueMap<T>>>& tokens);

    const bool addElecID_;
    typedef std::pair<std::string, edm::InputTag> NameTag;
    std::vector<NameTag> elecIDSrcs_;
    std::vector<edm::EDGetTokenT<edm::ValueMap<float>>> elecIDTokens_;

    // tools
    const GreaterByPt<Electron> pTComparator_;

    pat::helper::MultiIsolator isolator_;
    pat::helper::MultiIsolator::IsolationValuePairs isolatorTmpStorage_;  // better here than recreate at each event
    IsolationLabels isoDepositLabels_;
    std::vector<edm::EDGetTokenT<edm::ValueMap<IsoDeposit>>> isoDepositTokens_;
    IsolationLabels isolationValueLabels_;
    std::vector<edm::EDGetTokenT<edm::ValueMap<double>>> isolationValueTokens_;
    IsolationLabels isolationValueLabelsNoPFId_;
    std::vector<edm::EDGetTokenT<edm::ValueMap<double>>> isolationValueNoPFIdTokens_;

    const bool addEfficiencies_;
    pat::helper::EfficiencyLoader efficiencyLoader_;

    const bool addResolutions_;
    pat::helper::KinResolutionsLoader resolutionLoader_;

    const bool useUserData_;
    //PUPPI isolation tokens
    edm::EDGetTokenT<edm::ValueMap<float>> PUPPIIsolation_charged_hadrons_;
    edm::EDGetTokenT<edm::ValueMap<float>> PUPPIIsolation_neutral_hadrons_;
    edm::EDGetTokenT<edm::ValueMap<float>> PUPPIIsolation_photons_;
    //PUPPINoLeptons isolation tokens
    edm::EDGetTokenT<edm::ValueMap<float>> PUPPINoLeptonsIsolation_charged_hadrons_;
    edm::EDGetTokenT<edm::ValueMap<float>> PUPPINoLeptonsIsolation_neutral_hadrons_;
    edm::EDGetTokenT<edm::ValueMap<float>> PUPPINoLeptonsIsolation_photons_;
    pat::PATUserDataHelper<pat::Electron> userDataHelper_;

    const edm::ESGetToken<CaloTopology, CaloTopologyRecord> ecalTopologyToken_;
    const edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> trackBuilderToken_;

    const CaloTopology* ecalTopology_;
  };
}  // namespace pat

template <typename T>
void pat::PATElectronProducer::readIsolationLabels(const edm::ParameterSet& iConfig,
                                                   const char* psetName,
                                                   pat::PATElectronProducer::IsolationLabels& labels,
                                                   std::vector<edm::EDGetTokenT<edm::ValueMap<T>>>& tokens) {
  labels.clear();

  if (iConfig.exists(psetName)) {
    edm::ParameterSet depconf = iConfig.getParameter<edm::ParameterSet>(psetName);

    if (depconf.exists("tracker"))
      labels.push_back(std::make_pair(pat::TrackIso, depconf.getParameter<edm::InputTag>("tracker")));
    if (depconf.exists("ecal"))
      labels.push_back(std::make_pair(pat::EcalIso, depconf.getParameter<edm::InputTag>("ecal")));
    if (depconf.exists("hcal"))
      labels.push_back(std::make_pair(pat::HcalIso, depconf.getParameter<edm::InputTag>("hcal")));
    if (depconf.exists("pfAllParticles")) {
      labels.push_back(std::make_pair(pat::PfAllParticleIso, depconf.getParameter<edm::InputTag>("pfAllParticles")));
    }
    if (depconf.exists("pfChargedHadrons")) {
      labels.push_back(
          std::make_pair(pat::PfChargedHadronIso, depconf.getParameter<edm::InputTag>("pfChargedHadrons")));
    }
    if (depconf.exists("pfChargedAll")) {
      labels.push_back(std::make_pair(pat::PfChargedAllIso, depconf.getParameter<edm::InputTag>("pfChargedAll")));
    }
    if (depconf.exists("pfPUChargedHadrons")) {
      labels.push_back(
          std::make_pair(pat::PfPUChargedHadronIso, depconf.getParameter<edm::InputTag>("pfPUChargedHadrons")));
    }
    if (depconf.exists("pfNeutralHadrons")) {
      labels.push_back(
          std::make_pair(pat::PfNeutralHadronIso, depconf.getParameter<edm::InputTag>("pfNeutralHadrons")));
    }
    if (depconf.exists("pfPhotons")) {
      labels.push_back(std::make_pair(pat::PfGammaIso, depconf.getParameter<edm::InputTag>("pfPhotons")));
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
  tokens = edm::vector_transform(
      labels, [this](IsolationLabel const& label) { return consumes<edm::ValueMap<T>>(label.second); });
}

using namespace pat;
using namespace std;

PATElectronProducer::PATElectronProducer(const edm::ParameterSet& iConfig)
    :  // general configurables
      electronToken_(consumes<edm::View<reco::GsfElectron>>(iConfig.getParameter<edm::InputTag>("electronSource"))),
      hConversionsToken_(consumes<reco::ConversionCollection>(edm::InputTag("allConversions"))),
      embedGsfElectronCore_(iConfig.getParameter<bool>("embedGsfElectronCore")),
      embedGsfTrack_(iConfig.getParameter<bool>("embedGsfTrack")),
      embedSuperCluster_(iConfig.getParameter<bool>("embedSuperCluster")),
      embedPflowSuperCluster_(iConfig.getParameter<bool>("embedPflowSuperCluster")),
      embedSeedCluster_(iConfig.getParameter<bool>("embedSeedCluster")),
      embedBasicClusters_(iConfig.getParameter<bool>("embedBasicClusters")),
      embedPreshowerClusters_(iConfig.getParameter<bool>("embedPreshowerClusters")),
      embedPflowBasicClusters_(iConfig.getParameter<bool>("embedPflowBasicClusters")),
      embedPflowPreshowerClusters_(iConfig.getParameter<bool>("embedPflowPreshowerClusters")),
      embedTrack_(iConfig.getParameter<bool>("embedTrack")),
      addGenMatch_(iConfig.getParameter<bool>("addGenMatch")),
      embedGenMatch_(addGenMatch_ ? iConfig.getParameter<bool>("embedGenMatch") : false),
      embedRecHits_(iConfig.getParameter<bool>("embedRecHits")),
      // pflow configurables
      useParticleFlow_(iConfig.getParameter<bool>("useParticleFlow")),
      usePfCandidateMultiMap_(iConfig.getParameter<bool>("usePfCandidateMultiMap")),
      pfElecToken_(!usePfCandidateMultiMap_
                       ? consumes<reco::PFCandidateCollection>(iConfig.getParameter<edm::InputTag>("pfElectronSource"))
                       : edm::EDGetTokenT<reco::PFCandidateCollection>()),
      pfCandidateMapToken_(!usePfCandidateMultiMap_ ? mayConsume<edm::ValueMap<reco::PFCandidatePtr>>(
                                                          iConfig.getParameter<edm::InputTag>("pfCandidateMap"))
                                                    : edm::EDGetTokenT<edm::ValueMap<reco::PFCandidatePtr>>()),
      pfCandidateMultiMapToken_(usePfCandidateMultiMap_
                                    ? consumes<edm::ValueMap<std::vector<reco::PFCandidateRef>>>(
                                          iConfig.getParameter<edm::InputTag>("pfCandidateMultiMap"))
                                    : edm::EDGetTokenT<edm::ValueMap<std::vector<reco::PFCandidateRef>>>()),
      embedPFCandidate_(iConfig.getParameter<bool>("embedPFCandidate")),
      // mva input variables
      addMVAVariables_(iConfig.getParameter<bool>("addMVAVariables")),
      reducedBarrelRecHitCollection_(iConfig.getParameter<edm::InputTag>("reducedBarrelRecHitCollection")),
      reducedBarrelRecHitCollectionToken_(mayConsume<EcalRecHitCollection>(reducedBarrelRecHitCollection_)),
      reducedEndcapRecHitCollection_(iConfig.getParameter<edm::InputTag>("reducedEndcapRecHitCollection")),
      reducedEndcapRecHitCollectionToken_(mayConsume<EcalRecHitCollection>(reducedEndcapRecHitCollection_)),
      ecalClusterToolsESGetTokens_{consumesCollector()},
      // PFCluster Isolation maps
      addPFClusterIso_(iConfig.getParameter<bool>("addPFClusterIso")),
      addPuppiIsolation_(iConfig.getParameter<bool>("addPuppiIsolation")),
      ecalPFClusterIsoT_(consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("ecalPFClusterIsoMap"))),
      hcalPFClusterIsoT_(consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("hcalPFClusterIsoMap"))),
      // embed high level selection variables?
      embedHighLevelSelection_(iConfig.getParameter<bool>("embedHighLevelSelection")),
      beamLineToken_(consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamLineSrc"))),
      pvToken_(mayConsume<std::vector<reco::Vertex>>(iConfig.getParameter<edm::InputTag>("pvSrc"))),
      addElecID_(iConfig.getParameter<bool>("addElectronID")),
      pTComparator_(),
      isolator_(iConfig.getParameter<edm::ParameterSet>("userIsolation"), consumesCollector(), false),
      addEfficiencies_(iConfig.getParameter<bool>("addEfficiencies")),
      addResolutions_(iConfig.getParameter<bool>("addResolutions")),
      useUserData_(iConfig.exists("userData")),
      ecalTopologyToken_{esConsumes()},
      trackBuilderToken_{esConsumes(edm::ESInputTag("", "TransientTrackBuilder"))} {
  // MC matching configurables (scheduled mode)

  if (addGenMatch_) {
    genMatchTokens_.push_back(consumes<edm::Association<reco::GenParticleCollection>>(
        iConfig.getParameter<edm::InputTag>("genParticleMatch")));
  }
  // resolution configurables
  if (addResolutions_) {
    resolutionLoader_ =
        pat::helper::KinResolutionsLoader(iConfig.getParameter<edm::ParameterSet>("resolutions"), consumesCollector());
  }
  if (addPuppiIsolation_) {
    //puppi
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
  // electron ID configurables
  if (addElecID_) {
    // it might be a single electron ID
    if (iConfig.existsAs<edm::InputTag>("electronIDSource")) {
      elecIDSrcs_.push_back(NameTag("", iConfig.getParameter<edm::InputTag>("electronIDSource")));
    }
    // or there might be many of them
    if (iConfig.existsAs<edm::ParameterSet>("electronIDSources")) {
      // please don't configure me twice
      if (!elecIDSrcs_.empty()) {
        throw cms::Exception("Configuration")
            << "PATElectronProducer: you can't specify both 'electronIDSource' and 'electronIDSources'\n";
      }
      // read the different electron ID names
      edm::ParameterSet idps = iConfig.getParameter<edm::ParameterSet>("electronIDSources");
      std::vector<std::string> names = idps.getParameterNamesForType<edm::InputTag>();
      for (std::vector<std::string>::const_iterator it = names.begin(), ed = names.end(); it != ed; ++it) {
        elecIDSrcs_.push_back(NameTag(*it, idps.getParameter<edm::InputTag>(*it)));
      }
    }
    // but in any case at least once
    if (elecIDSrcs_.empty()) {
      throw cms::Exception("Configuration")
          << "PATElectronProducer: id addElectronID is true, you must specify either:\n"
          << "\tInputTag electronIDSource = <someTag>\n"
          << "or\n"
          << "\tPSet electronIDSources = { \n"
          << "\t\tInputTag <someName> = <someTag>   // as many as you want \n "
          << "\t}\n";
    }
  }
  elecIDTokens_ = edm::vector_transform(
      elecIDSrcs_, [this](NameTag const& tag) { return mayConsume<edm::ValueMap<float>>(tag.second); });
  // construct resolution calculator

  //   // IsoDeposit configurables
  //   if (iConfig.exists("isoDeposits")) {
  //      edm::ParameterSet depconf = iConfig.getParameter<edm::ParameterSet>("isoDeposits");
  //      if (depconf.exists("tracker")) isoDepositLabels_.push_back(std::make_pair(TrackerIso, depconf.getParameter<edm::InputTag>("tracker")));
  //      if (depconf.exists("ecal"))    isoDepositLabels_.push_back(std::make_pair(ECalIso, depconf.getParameter<edm::InputTag>("ecal")));
  //      if (depconf.exists("hcal"))    isoDepositLabels_.push_back(std::make_pair(HCalIso, depconf.getParameter<edm::InputTag>("hcal")));

  //      if (depconf.exists("user")) {
  //         std::vector<edm::InputTag> userdeps = depconf.getParameter<std::vector<edm::InputTag> >("user");
  //         std::vector<edm::InputTag>::const_iterator it = userdeps.begin(), ed = userdeps.end();
  //         int key = UserBaseIso;
  //         for ( ; it != ed; ++it, ++key) {
  //             isoDepositLabels_.push_back(std::make_pair(IsolationKeys(key), *it));
  //         }
  //      }
  //   }
  //   isoDepositTokens_ = edm::vector_transform(isoDepositLabels_, [this](std::pair<IsolationKeys,edm::InputTag> const & label){return consumes<edm::ValueMap<IsoDeposit> >(label.second);});

  // for mini-iso
  computeMiniIso_ = iConfig.getParameter<bool>("computeMiniIso");
  miniIsoParamsE_ = iConfig.getParameter<std::vector<double>>("miniIsoParamsE");
  miniIsoParamsB_ = iConfig.getParameter<std::vector<double>>("miniIsoParamsB");
  if (computeMiniIso_ && (miniIsoParamsE_.size() != 9 || miniIsoParamsB_.size() != 9)) {
    throw cms::Exception("ParameterError") << "miniIsoParams must have exactly 9 elements.\n";
  }
  if (computeMiniIso_)
    pcToken_ = consumes<pat::PackedCandidateCollection>(iConfig.getParameter<edm::InputTag>("pfCandsForMiniIso"));

  // read isoDeposit labels, for direct embedding
  readIsolationLabels(iConfig, "isoDeposits", isoDepositLabels_, isoDepositTokens_);
  // read isolation value labels, for direct embedding
  readIsolationLabels(iConfig, "isolationValues", isolationValueLabels_, isolationValueTokens_);
  // read isolation value labels for non PF identified electron, for direct embedding
  readIsolationLabels(iConfig, "isolationValuesNoPFId", isolationValueLabelsNoPFId_, isolationValueNoPFIdTokens_);
  // Efficiency configurables
  if (addEfficiencies_) {
    efficiencyLoader_ =
        pat::helper::EfficiencyLoader(iConfig.getParameter<edm::ParameterSet>("efficiencies"), consumesCollector());
  }
  // Check to see if the user wants to add user data
  if (useUserData_) {
    userDataHelper_ =
        PATUserDataHelper<Electron>(iConfig.getParameter<edm::ParameterSet>("userData"), consumesCollector());
  }

  // consistency check
  if (useParticleFlow_ && usePfCandidateMultiMap_)
    throw cms::Exception("Configuration", "usePfCandidateMultiMap not supported when useParticleFlow is set to true");

  // produces vector of muons
  produces<std::vector<Electron>>();
}

PATElectronProducer::~PATElectronProducer() {}

void PATElectronProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // switch off embedding (in unschedules mode)
  if (iEvent.isRealData()) {
    addGenMatch_ = false;
    embedGenMatch_ = false;
  }

  ecalTopology_ = &iSetup.getData(ecalTopologyToken_);

  // Get the collection of electrons from the event
  edm::Handle<edm::View<reco::GsfElectron>> electrons;
  iEvent.getByToken(electronToken_, electrons);

  edm::Handle<PackedCandidateCollection> pc;
  if (computeMiniIso_)
    iEvent.getByToken(pcToken_, pc);

  // for additional mva variables
  edm::InputTag reducedEBRecHitCollection(string("reducedEcalRecHitsEB"));
  edm::InputTag reducedEERecHitCollection(string("reducedEcalRecHitsEE"));
  //EcalClusterLazyTools lazyTools(iEvent, iSetup, reducedEBRecHitCollection, reducedEERecHitCollection);
  EcalClusterLazyTools lazyTools(iEvent,
                                 ecalClusterToolsESGetTokens_.get(iSetup),
                                 reducedBarrelRecHitCollectionToken_,
                                 reducedEndcapRecHitCollectionToken_);

  // for conversion veto selection
  edm::Handle<reco::ConversionCollection> hConversions;
  iEvent.getByToken(hConversionsToken_, hConversions);

  // Get the ESHandle for the transient track builder, if needed for
  // high level selection embedding
  edm::ESHandle<TransientTrackBuilder> trackBuilder;

  if (isolator_.enabled())
    isolator_.beginEvent(iEvent, iSetup);

  if (efficiencyLoader_.enabled())
    efficiencyLoader_.newEvent(iEvent);
  if (resolutionLoader_.enabled())
    resolutionLoader_.newEvent(iEvent, iSetup);

  IsoDepositMaps deposits(isoDepositTokens_.size());
  for (size_t j = 0, nd = isoDepositTokens_.size(); j < nd; ++j) {
    iEvent.getByToken(isoDepositTokens_[j], deposits[j]);
  }

  IsolationValueMaps isolationValues(isolationValueTokens_.size());
  for (size_t j = 0; j < isolationValueTokens_.size(); ++j) {
    iEvent.getByToken(isolationValueTokens_[j], isolationValues[j]);
  }

  IsolationValueMaps isolationValuesNoPFId(isolationValueNoPFIdTokens_.size());
  for (size_t j = 0; j < isolationValueNoPFIdTokens_.size(); ++j) {
    iEvent.getByToken(isolationValueNoPFIdTokens_[j], isolationValuesNoPFId[j]);
  }

  // prepare the MC matching
  GenAssociations genMatches(genMatchTokens_.size());
  if (addGenMatch_) {
    for (size_t j = 0, nd = genMatchTokens_.size(); j < nd; ++j) {
      iEvent.getByToken(genMatchTokens_[j], genMatches[j]);
    }
  }

  // prepare ID extraction
  std::vector<edm::Handle<edm::ValueMap<float>>> idhandles;
  std::vector<pat::Electron::IdPair> ids;
  if (addElecID_) {
    idhandles.resize(elecIDSrcs_.size());
    ids.resize(elecIDSrcs_.size());
    for (size_t i = 0; i < elecIDSrcs_.size(); ++i) {
      iEvent.getByToken(elecIDTokens_[i], idhandles[i]);
      ids[i].first = elecIDSrcs_[i].first;
    }
  }

  // prepare the high level selection:
  // needs beamline
  reco::TrackBase::Point beamPoint(0, 0, 0);
  reco::Vertex primaryVertex;
  reco::BeamSpot beamSpot;
  bool beamSpotIsValid = false;
  bool primaryVertexIsValid = false;

  // Get the beamspot
  edm::Handle<reco::BeamSpot> beamSpotHandle;
  iEvent.getByToken(beamLineToken_, beamSpotHandle);

  if (embedHighLevelSelection_) {
    // Get the primary vertex
    edm::Handle<std::vector<reco::Vertex>> pvHandle;
    iEvent.getByToken(pvToken_, pvHandle);

    // This is needed by the IPTools methods from the tracking group
    trackBuilder = iSetup.getHandle(trackBuilderToken_);

    if (pvHandle.isValid() && !pvHandle->empty()) {
      primaryVertex = pvHandle->at(0);
      primaryVertexIsValid = true;
    } else {
      edm::LogError("DataNotAvailable")
          << "No primary vertex available from EventSetup, not adding high level selection \n";
    }
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

  std::vector<Electron>* patElectrons = new std::vector<Electron>();

  if (useParticleFlow_) {
    edm::Handle<reco::PFCandidateCollection> pfElectrons;
    iEvent.getByToken(pfElecToken_, pfElectrons);
    unsigned index = 0;

    for (reco::PFCandidateConstIterator i = pfElectrons->begin(); i != pfElectrons->end(); ++i, ++index) {
      reco::PFCandidateRef pfRef(pfElectrons, index);
      reco::PFCandidatePtr ptrToPFElectron(pfElectrons, index);
      //       reco::CandidateBaseRef pfBaseRef( pfRef );

      reco::GsfTrackRef PfTk = i->gsfTrackRef();

      bool Matched = false;
      bool MatchedToAmbiguousGsfTrack = false;
      for (edm::View<reco::GsfElectron>::const_iterator itElectron = electrons->begin(); itElectron != electrons->end();
           ++itElectron) {
        unsigned int idx = itElectron - electrons->begin();
        auto elePtr = electrons->ptrAt(idx);
        if (Matched || MatchedToAmbiguousGsfTrack)
          continue;

        reco::GsfTrackRef EgTk = itElectron->gsfTrack();

        if (itElectron->gsfTrack() == i->gsfTrackRef()) {
          Matched = true;
        } else {
          for (auto const& it : itElectron->ambiguousGsfTracks()) {
            MatchedToAmbiguousGsfTrack |= (bool)(i->gsfTrackRef() == it);
          }
        }

        if (Matched || MatchedToAmbiguousGsfTrack) {
          // ptr needed for finding the matched gen particle
          reco::CandidatePtr ptrToGsfElectron(electrons, idx);

          // ref to base needed for the construction of the pat object
          const edm::RefToBase<reco::GsfElectron>& elecsRef = electrons->refAt(idx);
          Electron anElectron(elecsRef);
          anElectron.setPFCandidateRef(pfRef);
          if (addPuppiIsolation_) {
            anElectron.setIsolationPUPPI((*PUPPIIsolation_charged_hadrons)[elePtr],
                                         (*PUPPIIsolation_neutral_hadrons)[elePtr],
                                         (*PUPPIIsolation_photons)[elePtr]);
            anElectron.setIsolationPUPPINoLeptons((*PUPPINoLeptonsIsolation_charged_hadrons)[elePtr],
                                                  (*PUPPINoLeptonsIsolation_neutral_hadrons)[elePtr],
                                                  (*PUPPINoLeptonsIsolation_photons)[elePtr]);
          } else {
            anElectron.setIsolationPUPPI(-999., -999., -999.);
            anElectron.setIsolationPUPPINoLeptons(-999., -999., -999.);
          }

          //it should be always true when particleFlow electrons are used.
          anElectron.setIsPF(true);

          if (embedPFCandidate_)
            anElectron.embedPFCandidate();

          if (useUserData_) {
            userDataHelper_.add(anElectron, iEvent, iSetup);
          }

          double ip3d = -999;  // for mva variable

          // embed high level selection
          if (embedHighLevelSelection_) {
            // get the global track
            const reco::GsfTrackRef& track = PfTk;

            // Make sure the collection it points to is there
            if (track.isNonnull() && track.isAvailable()) {
              reco::TransientTrack tt = trackBuilder->build(track);
              embedHighLevel(anElectron, track, tt, primaryVertex, primaryVertexIsValid, beamSpot, beamSpotIsValid);

              std::pair<bool, Measurement1D> ip3dpv = IPTools::absoluteImpactParameter3D(tt, primaryVertex);
              ip3d = ip3dpv.second.value();  // for mva variable
            }
          }

          //Electron Id

          if (addElecID_) {
            //STANDARD EL ID
            for (size_t i = 0; i < elecIDSrcs_.size(); ++i) {
              ids[i].second = (*idhandles[i])[elecsRef];
            }
            //SPECIFIC PF ID
            ids.push_back(std::make_pair("pf_evspi", pfRef->mva_e_pi()));
            ids.push_back(std::make_pair("pf_evsmu", pfRef->mva_e_mu()));
            anElectron.setElectronIDs(ids);
          }

          if (addMVAVariables_) {
            // add missing mva variables
            const auto& vCov = lazyTools.localCovariances(*(itElectron->superCluster()->seed()));
            anElectron.setMvaVariables(vCov[1], ip3d);
          }
          // PFClusterIso
          if (addPFClusterIso_) {
            // Get PFCluster Isolation
            edm::Handle<edm::ValueMap<float>> ecalPFClusterIsoMapH;
            iEvent.getByToken(ecalPFClusterIsoT_, ecalPFClusterIsoMapH);
            edm::Handle<edm::ValueMap<float>> hcalPFClusterIsoMapH;
            iEvent.getByToken(hcalPFClusterIsoT_, hcalPFClusterIsoMapH);
            reco::GsfElectron::PflowIsolationVariables newPFIsol = anElectron.pfIsolationVariables();
            newPFIsol.sumEcalClusterEt = (*ecalPFClusterIsoMapH)[elecsRef];
            newPFIsol.sumHcalClusterEt = (*hcalPFClusterIsoMapH)[elecsRef];
            anElectron.setPfIsolationVariables(newPFIsol);
          }

          std::vector<DetId> selectedCells;
          bool barrel = itElectron->isEB();
          //loop over sub clusters
          if (embedBasicClusters_) {
            for (reco::CaloCluster_iterator clusIt = itElectron->superCluster()->clustersBegin();
                 clusIt != itElectron->superCluster()->clustersEnd();
                 ++clusIt) {
              //get seed (max energy xtal)
              DetId seed = lazyTools.getMaximum(**clusIt).first;
              //get all xtals in 5x5 window around the seed
              std::vector<DetId> dets5x5 =
                  (barrel) ? ecalTopology_->getSubdetectorTopology(DetId::Ecal, EcalBarrel)->getWindow(seed, 5, 5)
                           : ecalTopology_->getSubdetectorTopology(DetId::Ecal, EcalEndcap)->getWindow(seed, 5, 5);
              selectedCells.insert(selectedCells.end(), dets5x5.begin(), dets5x5.end());

              //get all xtals belonging to cluster
              for (const std::pair<DetId, float>& hit : (*clusIt)->hitsAndFractions()) {
                selectedCells.push_back(hit.first);
              }
            }
          }

          if (embedPflowBasicClusters_ && itElectron->parentSuperCluster().isNonnull()) {
            for (reco::CaloCluster_iterator clusIt = itElectron->parentSuperCluster()->clustersBegin();
                 clusIt != itElectron->parentSuperCluster()->clustersEnd();
                 ++clusIt) {
              //get seed (max energy xtal)
              DetId seed = lazyTools.getMaximum(**clusIt).first;
              //get all xtals in 5x5 window around the seed
              std::vector<DetId> dets5x5 =
                  (barrel) ? ecalTopology_->getSubdetectorTopology(DetId::Ecal, EcalBarrel)->getWindow(seed, 5, 5)
                           : ecalTopology_->getSubdetectorTopology(DetId::Ecal, EcalEndcap)->getWindow(seed, 5, 5);
              selectedCells.insert(selectedCells.end(), dets5x5.begin(), dets5x5.end());

              //get all xtals belonging to cluster
              for (const std::pair<DetId, float>& hit : (*clusIt)->hitsAndFractions()) {
                selectedCells.push_back(hit.first);
              }
            }
          }

          //remove duplicates
          std::sort(selectedCells.begin(), selectedCells.end());
          std::unique(selectedCells.begin(), selectedCells.end());

          // Retrieve the corresponding RecHits

          edm::Handle<EcalRecHitCollection> rechitsH;
          if (barrel)
            iEvent.getByToken(reducedBarrelRecHitCollectionToken_, rechitsH);
          else
            iEvent.getByToken(reducedEndcapRecHitCollectionToken_, rechitsH);

          EcalRecHitCollection selectedRecHits;
          const EcalRecHitCollection* recHits = rechitsH.product();

          unsigned nSelectedCells = selectedCells.size();
          for (unsigned icell = 0; icell < nSelectedCells; ++icell) {
            EcalRecHitCollection::const_iterator it = recHits->find(selectedCells[icell]);
            if (it != recHits->end()) {
              selectedRecHits.push_back(*it);
            }
          }
          selectedRecHits.sort();
          if (embedRecHits_)
            anElectron.embedRecHits(&selectedRecHits);

          // set conversion veto selection
          bool passconversionveto = false;
          if (hConversions.isValid()) {
            // this is recommended method
            passconversionveto =
                !ConversionTools::hasMatchedConversion(*itElectron, *hConversions, beamSpotHandle->position());
          } else {
            // use missing hits without vertex fit method
            passconversionveto =
                itElectron->gsfTrack()->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS) < 1;
          }

          anElectron.setPassConversionVeto(passconversionveto);

          // 	  fillElectron(anElectron,elecsRef,pfBaseRef,
          // 		       genMatches, deposits, isolationValues);

          //COLIN small warning !
          // we are currently choosing to take the 4-momentum of the PFCandidate;
          // the momentum of the GsfElectron is saved though
          // we must therefore match the GsfElectron.
          // because of this, we should not change the source of the electron matcher
          // to the collection of PFElectrons in the python configuration
          // I don't know what to do with the efficiencyLoader, since I don't know
          // what this class is for.
          fillElectron2(
              anElectron, ptrToPFElectron, ptrToGsfElectron, ptrToGsfElectron, genMatches, deposits, isolationValues);

          //COLIN need to use fillElectron2 in the non-pflow case as well, and to test it.

          if (computeMiniIso_)
            setElectronMiniIso(anElectron, pc.product());

          patElectrons->push_back(anElectron);
        }
      }
      //if( !Matched && !MatchedToAmbiguousGsfTrack) std::cout << "!!!!A pf electron could not be matched to a gsf!!!!"  << std::endl;
    }
  }

  else {
    edm::Handle<reco::PFCandidateCollection> pfElectrons;
    edm::Handle<edm::ValueMap<reco::PFCandidatePtr>> ValMapH;
    edm::Handle<edm::ValueMap<std::vector<reco::PFCandidateRef>>> ValMultiMapH;
    bool pfCandsPresent = false, valMapPresent = false;
    if (usePfCandidateMultiMap_) {
      iEvent.getByToken(pfCandidateMultiMapToken_, ValMultiMapH);
    } else {
      pfCandsPresent = iEvent.getByToken(pfElecToken_, pfElectrons);
      valMapPresent = iEvent.getByToken(pfCandidateMapToken_, ValMapH);
    }

    for (edm::View<reco::GsfElectron>::const_iterator itElectron = electrons->begin(); itElectron != electrons->end();
         ++itElectron) {
      // construct the Electron from the ref -> save ref to original object
      //FIXME: looks like a lot of instances could be turned into const refs
      unsigned int idx = itElectron - electrons->begin();
      edm::RefToBase<reco::GsfElectron> elecsRef = electrons->refAt(idx);
      reco::CandidateBaseRef elecBaseRef(elecsRef);
      Electron anElectron(elecsRef);
      auto elePtr = electrons->ptrAt(idx);

      // Is this GsfElectron also identified as an e- in the particle flow?
      bool pfId = false;

      if (usePfCandidateMultiMap_) {
        for (const reco::PFCandidateRef& pf : (*ValMultiMapH)[elePtr]) {
          if (pf->particleId() == reco::PFCandidate::e) {
            pfId = true;
            anElectron.setPFCandidateRef(pf);
            break;
          }
        }
      } else if (pfCandsPresent) {
        // PF electron collection not available.
        const reco::GsfTrackRef& trkRef = itElectron->gsfTrack();
        int index = 0;
        for (reco::PFCandidateConstIterator ie = pfElectrons->begin(); ie != pfElectrons->end(); ++ie, ++index) {
          if (ie->particleId() != reco::PFCandidate::e)
            continue;
          const reco::GsfTrackRef& pfTrkRef = ie->gsfTrackRef();
          if (trkRef == pfTrkRef) {
            pfId = true;
            reco::PFCandidateRef pfRef(pfElectrons, index);
            anElectron.setPFCandidateRef(pfRef);
            break;
          }
        }
      } else if (valMapPresent) {
        // use value map if PF collection not available
        const edm::ValueMap<reco::PFCandidatePtr>& myValMap(*ValMapH);
        // Get the PFCandidate
        const reco::PFCandidatePtr& pfElePtr(myValMap[elecsRef]);
        pfId = pfElePtr.isNonnull();
      }
      // set PFId function
      anElectron.setIsPF(pfId);

      // add resolution info

      // Isolation
      if (isolator_.enabled()) {
        isolator_.fill(*electrons, idx, isolatorTmpStorage_);
        typedef pat::helper::MultiIsolator::IsolationValuePairs IsolationValuePairs;
        // better to loop backwards, so the vector is resized less times
        for (IsolationValuePairs::const_reverse_iterator it = isolatorTmpStorage_.rbegin(),
                                                         ed = isolatorTmpStorage_.rend();
             it != ed;
             ++it) {
          anElectron.setIsolation(it->first, it->second);
        }
      }

      for (size_t j = 0, nd = deposits.size(); j < nd; ++j) {
        anElectron.setIsoDeposit(isoDepositLabels_[j].first, (*deposits[j])[elecsRef]);
      }

      // add electron ID info
      if (addElecID_) {
        for (size_t i = 0; i < elecIDSrcs_.size(); ++i) {
          ids[i].second = (*idhandles[i])[elecsRef];
        }
        anElectron.setElectronIDs(ids);
      }

      if (useUserData_) {
        userDataHelper_.add(anElectron, iEvent, iSetup);
      }

      double ip3d = -999;  //for mva variable

      // embed high level selection
      if (embedHighLevelSelection_) {
        // get the global track
        reco::GsfTrackRef track = itElectron->gsfTrack();

        // Make sure the collection it points to is there
        if (track.isNonnull() && track.isAvailable()) {
          reco::TransientTrack tt = trackBuilder->build(track);
          embedHighLevel(anElectron, track, tt, primaryVertex, primaryVertexIsValid, beamSpot, beamSpotIsValid);

          std::pair<bool, Measurement1D> ip3dpv = IPTools::absoluteImpactParameter3D(tt, primaryVertex);
          ip3d = ip3dpv.second.value();  // for mva variable
        }
      }

      if (addMVAVariables_) {
        // add mva variables
        const auto& vCov = lazyTools.localCovariances(*(itElectron->superCluster()->seed()));
        anElectron.setMvaVariables(vCov[1], ip3d);
      }

      // PFCluster Isolation
      if (addPFClusterIso_) {
        // Get PFCluster Isolation
        edm::Handle<edm::ValueMap<float>> ecalPFClusterIsoMapH;
        iEvent.getByToken(ecalPFClusterIsoT_, ecalPFClusterIsoMapH);
        edm::Handle<edm::ValueMap<float>> hcalPFClusterIsoMapH;
        iEvent.getByToken(hcalPFClusterIsoT_, hcalPFClusterIsoMapH);
        reco::GsfElectron::PflowIsolationVariables newPFIsol = anElectron.pfIsolationVariables();
        newPFIsol.sumEcalClusterEt = (*ecalPFClusterIsoMapH)[elecsRef];
        newPFIsol.sumHcalClusterEt = (*hcalPFClusterIsoMapH)[elecsRef];
        anElectron.setPfIsolationVariables(newPFIsol);
      }

      if (addPuppiIsolation_) {
        anElectron.setIsolationPUPPI((*PUPPIIsolation_charged_hadrons)[elePtr],
                                     (*PUPPIIsolation_neutral_hadrons)[elePtr],
                                     (*PUPPIIsolation_photons)[elePtr]);
        anElectron.setIsolationPUPPINoLeptons((*PUPPINoLeptonsIsolation_charged_hadrons)[elePtr],
                                              (*PUPPINoLeptonsIsolation_neutral_hadrons)[elePtr],
                                              (*PUPPINoLeptonsIsolation_photons)[elePtr]);
      } else {
        anElectron.setIsolationPUPPI(-999., -999., -999.);
        anElectron.setIsolationPUPPINoLeptons(-999., -999., -999.);
      }

      std::vector<DetId> selectedCells;
      bool barrel = itElectron->isEB();
      //loop over sub clusters
      if (embedBasicClusters_) {
        for (reco::CaloCluster_iterator clusIt = itElectron->superCluster()->clustersBegin();
             clusIt != itElectron->superCluster()->clustersEnd();
             ++clusIt) {
          //get seed (max energy xtal)
          DetId seed = lazyTools.getMaximum(**clusIt).first;
          //get all xtals in 5x5 window around the seed
          std::vector<DetId> dets5x5 =
              (barrel) ? ecalTopology_->getSubdetectorTopology(DetId::Ecal, EcalBarrel)->getWindow(seed, 5, 5)
                       : ecalTopology_->getSubdetectorTopology(DetId::Ecal, EcalEndcap)->getWindow(seed, 5, 5);
          selectedCells.insert(selectedCells.end(), dets5x5.begin(), dets5x5.end());

          //get all xtals belonging to cluster
          for (const std::pair<DetId, float>& hit : (*clusIt)->hitsAndFractions()) {
            selectedCells.push_back(hit.first);
          }
        }
      }

      if (embedPflowBasicClusters_ && itElectron->parentSuperCluster().isNonnull()) {
        for (reco::CaloCluster_iterator clusIt = itElectron->parentSuperCluster()->clustersBegin();
             clusIt != itElectron->parentSuperCluster()->clustersEnd();
             ++clusIt) {
          //get seed (max energy xtal)
          DetId seed = lazyTools.getMaximum(**clusIt).first;
          //get all xtals in 5x5 window around the seed
          std::vector<DetId> dets5x5 =
              (barrel) ? ecalTopology_->getSubdetectorTopology(DetId::Ecal, EcalBarrel)->getWindow(seed, 5, 5)
                       : ecalTopology_->getSubdetectorTopology(DetId::Ecal, EcalEndcap)->getWindow(seed, 5, 5);
          selectedCells.insert(selectedCells.end(), dets5x5.begin(), dets5x5.end());

          //get all xtals belonging to cluster
          for (const std::pair<DetId, float>& hit : (*clusIt)->hitsAndFractions()) {
            selectedCells.push_back(hit.first);
          }
        }
      }

      //remove duplicates
      std::sort(selectedCells.begin(), selectedCells.end());
      std::unique(selectedCells.begin(), selectedCells.end());

      // Retrieve the corresponding RecHits

      edm::Handle<EcalRecHitCollection> rechitsH;
      if (barrel)
        iEvent.getByToken(reducedBarrelRecHitCollectionToken_, rechitsH);
      else
        iEvent.getByToken(reducedEndcapRecHitCollectionToken_, rechitsH);

      EcalRecHitCollection selectedRecHits;
      const EcalRecHitCollection* recHits = rechitsH.product();

      unsigned nSelectedCells = selectedCells.size();
      for (unsigned icell = 0; icell < nSelectedCells; ++icell) {
        EcalRecHitCollection::const_iterator it = recHits->find(selectedCells[icell]);
        if (it != recHits->end()) {
          selectedRecHits.push_back(*it);
        }
      }
      selectedRecHits.sort();
      if (embedRecHits_)
        anElectron.embedRecHits(&selectedRecHits);

      // set conversion veto selection
      bool passconversionveto = false;
      if (hConversions.isValid()) {
        // this is recommended method
        passconversionveto =
            !ConversionTools::hasMatchedConversion(*itElectron, *hConversions, beamSpotHandle->position());
      } else {
        // use missing hits without vertex fit method
        passconversionveto =
            itElectron->gsfTrack()->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS) < 1;
      }
      anElectron.setPassConversionVeto(passconversionveto);

      // add sel to selected
      fillElectron(
          anElectron, elecsRef, elecBaseRef, genMatches, deposits, pfId, isolationValues, isolationValuesNoPFId);

      if (computeMiniIso_)
        setElectronMiniIso(anElectron, pc.product());

      patElectrons->push_back(anElectron);
    }
  }

  // sort electrons in pt
  std::sort(patElectrons->begin(), patElectrons->end(), pTComparator_);

  // add the electrons to the event output
  std::unique_ptr<std::vector<Electron>> ptr(patElectrons);
  iEvent.put(std::move(ptr));

  // clean up
  if (isolator_.enabled())
    isolator_.endEvent();
}

void PATElectronProducer::fillElectron(Electron& anElectron,
                                       const edm::RefToBase<reco::GsfElectron>& elecRef,
                                       const reco::CandidateBaseRef& baseRef,
                                       const GenAssociations& genMatches,
                                       const IsoDepositMaps& deposits,
                                       const bool pfId,
                                       const IsolationValueMaps& isolationValues,
                                       const IsolationValueMaps& isolationValuesNoPFId) const {
  //COLIN: might want to use the PFCandidate 4-mom. Which one is in use now?
  //   if (useParticleFlow_)
  //     aMuon.setP4( aMuon.pfCandidateRef()->p4() );

  //COLIN:
  //In the embedding case, the reference cannot be used to look into a value map.
  //therefore, one has to had the PFCandidateRef to this function, which becomes a bit
  //too much specific.

  // in fact, this function needs a baseref or ptr for genmatch
  // and a baseref or ptr for isodeposits and isolationvalues.
  // baseref is not needed
  // the ptrForIsolation and ptrForMatching should be defined upstream.

  // is the concrete elecRef needed for the efficiency loader? what is this loader?
  // how can we make it compatible with the particle flow electrons?

  if (embedGsfElectronCore_)
    anElectron.embedGsfElectronCore();
  if (embedGsfTrack_)
    anElectron.embedGsfTrack();
  if (embedSuperCluster_)
    anElectron.embedSuperCluster();
  if (embedPflowSuperCluster_)
    anElectron.embedPflowSuperCluster();
  if (embedSeedCluster_)
    anElectron.embedSeedCluster();
  if (embedBasicClusters_)
    anElectron.embedBasicClusters();
  if (embedPreshowerClusters_)
    anElectron.embedPreshowerClusters();
  if (embedPflowBasicClusters_)
    anElectron.embedPflowBasicClusters();
  if (embedPflowPreshowerClusters_)
    anElectron.embedPflowPreshowerClusters();
  if (embedTrack_)
    anElectron.embedTrack();

  // store the match to the generated final state muons
  if (addGenMatch_) {
    for (size_t i = 0, n = genMatches.size(); i < n; ++i) {
      if (useParticleFlow_) {
        reco::GenParticleRef genElectron = (*genMatches[i])[anElectron.pfCandidateRef()];
        anElectron.addGenParticleRef(genElectron);
      } else {
        reco::GenParticleRef genElectron = (*genMatches[i])[elecRef];
        anElectron.addGenParticleRef(genElectron);
      }
    }
    if (embedGenMatch_)
      anElectron.embedGenParticle();
  }

  if (efficiencyLoader_.enabled()) {
    efficiencyLoader_.setEfficiencies(anElectron, elecRef);
  }

  if (resolutionLoader_.enabled()) {
    resolutionLoader_.setResolutions(anElectron);
  }

  for (size_t j = 0, nd = deposits.size(); j < nd; ++j) {
    if (useParticleFlow_) {
      reco::PFCandidateRef pfcandref = anElectron.pfCandidateRef();
      assert(!pfcandref.isNull());
      reco::CandidatePtr source = pfcandref->sourceCandidatePtr(0);
      anElectron.setIsoDeposit(isoDepositLabels_[j].first, (*deposits[j])[source]);
    } else
      anElectron.setIsoDeposit(isoDepositLabels_[j].first, (*deposits[j])[elecRef]);
  }

  for (size_t j = 0; j < isolationValues.size(); ++j) {
    if (useParticleFlow_) {
      reco::CandidatePtr source = anElectron.pfCandidateRef()->sourceCandidatePtr(0);
      anElectron.setIsolation(isolationValueLabels_[j].first, (*isolationValues[j])[source]);
    } else if (pfId) {
      anElectron.setIsolation(isolationValueLabels_[j].first, (*isolationValues[j])[elecRef]);
    }
  }

  //for electrons not identified as PF electrons
  for (size_t j = 0; j < isolationValuesNoPFId.size(); ++j) {
    if (!pfId) {
      anElectron.setIsolation(isolationValueLabelsNoPFId_[j].first, (*isolationValuesNoPFId[j])[elecRef]);
    }
  }
}

void PATElectronProducer::fillElectron2(Electron& anElectron,
                                        const reco::CandidatePtr& candPtrForIsolation,
                                        const reco::CandidatePtr& candPtrForGenMatch,
                                        const reco::CandidatePtr& candPtrForLoader,
                                        const GenAssociations& genMatches,
                                        const IsoDepositMaps& deposits,
                                        const IsolationValueMaps& isolationValues) const {
  //COLIN/Florian: use the PFCandidate 4-mom.
  anElectron.setEcalDrivenMomentum(anElectron.p4());
  anElectron.setP4(anElectron.pfCandidateRef()->p4());

  // is the concrete elecRef needed for the efficiency loader? what is this loader?
  // how can we make it compatible with the particle flow electrons?

  if (embedGsfElectronCore_)
    anElectron.embedGsfElectronCore();
  if (embedGsfTrack_)
    anElectron.embedGsfTrack();
  if (embedSuperCluster_)
    anElectron.embedSuperCluster();
  if (embedPflowSuperCluster_)
    anElectron.embedPflowSuperCluster();
  if (embedSeedCluster_)
    anElectron.embedSeedCluster();
  if (embedBasicClusters_)
    anElectron.embedBasicClusters();
  if (embedPreshowerClusters_)
    anElectron.embedPreshowerClusters();
  if (embedPflowBasicClusters_)
    anElectron.embedPflowBasicClusters();
  if (embedPflowPreshowerClusters_)
    anElectron.embedPflowPreshowerClusters();
  if (embedTrack_)
    anElectron.embedTrack();

  // store the match to the generated final state muons

  if (addGenMatch_) {
    for (size_t i = 0, n = genMatches.size(); i < n; ++i) {
      reco::GenParticleRef genElectron = (*genMatches[i])[candPtrForGenMatch];
      anElectron.addGenParticleRef(genElectron);
    }
    if (embedGenMatch_)
      anElectron.embedGenParticle();
  }

  //COLIN what's this? does it have to be GsfElectron specific?
  if (efficiencyLoader_.enabled()) {
    efficiencyLoader_.setEfficiencies(anElectron, candPtrForLoader);
  }

  if (resolutionLoader_.enabled()) {
    resolutionLoader_.setResolutions(anElectron);
  }

  for (size_t j = 0, nd = deposits.size(); j < nd; ++j) {
    if (isoDepositLabels_[j].first == pat::TrackIso || isoDepositLabels_[j].first == pat::EcalIso ||
        isoDepositLabels_[j].first == pat::HcalIso || deposits[j]->contains(candPtrForGenMatch.id())) {
      anElectron.setIsoDeposit(isoDepositLabels_[j].first, (*deposits[j])[candPtrForGenMatch]);
    } else if (deposits[j]->contains(candPtrForIsolation.id())) {
      anElectron.setIsoDeposit(isoDepositLabels_[j].first, (*deposits[j])[candPtrForIsolation]);
    } else {
      anElectron.setIsoDeposit(isoDepositLabels_[j].first, (*deposits[j])[candPtrForIsolation->sourceCandidatePtr(0)]);
    }
  }

  for (size_t j = 0; j < isolationValues.size(); ++j) {
    if (isolationValueLabels_[j].first == pat::TrackIso || isolationValueLabels_[j].first == pat::EcalIso ||
        isolationValueLabels_[j].first == pat::HcalIso || isolationValues[j]->contains(candPtrForGenMatch.id())) {
      anElectron.setIsolation(isolationValueLabels_[j].first, (*isolationValues[j])[candPtrForGenMatch]);
    } else if (isolationValues[j]->contains(candPtrForIsolation.id())) {
      anElectron.setIsolation(isolationValueLabels_[j].first, (*isolationValues[j])[candPtrForIsolation]);
    } else {
      anElectron.setIsolation(isolationValueLabels_[j].first,
                              (*isolationValues[j])[candPtrForIsolation->sourceCandidatePtr(0)]);
    }
  }
}

void PATElectronProducer::setElectronMiniIso(Electron& anElectron, const PackedCandidateCollection* pc) {
  pat::PFIsolation miniiso;
  if (anElectron.isEE())
    miniiso = pat::getMiniPFIsolation(pc,
                                      anElectron.polarP4(),
                                      miniIsoParamsE_[0],
                                      miniIsoParamsE_[1],
                                      miniIsoParamsE_[2],
                                      miniIsoParamsE_[3],
                                      miniIsoParamsE_[4],
                                      miniIsoParamsE_[5],
                                      miniIsoParamsE_[6],
                                      miniIsoParamsE_[7],
                                      miniIsoParamsE_[8]);
  else
    miniiso = pat::getMiniPFIsolation(pc,
                                      anElectron.polarP4(),
                                      miniIsoParamsB_[0],
                                      miniIsoParamsB_[1],
                                      miniIsoParamsB_[2],
                                      miniIsoParamsB_[3],
                                      miniIsoParamsB_[4],
                                      miniIsoParamsB_[5],
                                      miniIsoParamsB_[6],
                                      miniIsoParamsB_[7],
                                      miniIsoParamsB_[8]);
  anElectron.setMiniPFIsolation(miniiso);
}

// ParameterSet description for module
void PATElectronProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription iDesc;
  iDesc.setComment("PAT electron producer module");

  // input source
  iDesc.add<edm::InputTag>("pfCandidateMap", edm::InputTag("no default"))->setComment("input collection");
  iDesc.add<edm::InputTag>("electronSource", edm::InputTag("no default"))->setComment("input collection");

  iDesc.ifValue(
      edm::ParameterDescription<bool>("addPFClusterIso", false, true),
      true >> (edm::ParameterDescription<edm::InputTag>(
                   "ecalPFClusterIsoMap", edm::InputTag("electronEcalPFClusterIsolationProducer"), true) and
               edm::ParameterDescription<edm::InputTag>(
                   "hcalPFClusterIsoMap", edm::InputTag("electronHcalPFClusterIsolationProducer"), true)) or
          false >> (edm::ParameterDescription<edm::InputTag>("ecalPFClusterIsoMap", edm::InputTag(""), true) and
                    edm::ParameterDescription<edm::InputTag>("hcalPFClusterIsoMap", edm::InputTag(""), true)));

  iDesc.ifValue(edm::ParameterDescription<bool>("addPuppiIsolation", false, true),
                true >> (edm::ParameterDescription<edm::InputTag>(
                             "puppiIsolationChargedHadrons",
                             edm::InputTag("egmElectronPUPPIIsolation", "h+-DR030-BarVeto000-EndVeto001"),
                             true) and
                         edm::ParameterDescription<edm::InputTag>(
                             "puppiIsolationNeutralHadrons",
                             edm::InputTag("egmElectronPUPPIIsolation", "h0-DR030-BarVeto000-EndVeto000"),
                             true) and
                         edm::ParameterDescription<edm::InputTag>(
                             "puppiIsolationPhotons",
                             edm::InputTag("egmElectronPUPPIIsolation", "gamma-DR030-BarVeto000-EndVeto008"),
                             true) and
                         edm::ParameterDescription<edm::InputTag>(
                             "puppiNoLeptonsIsolationChargedHadrons",
                             edm::InputTag("egmElectronPUPPINoLeptonsIsolation", "gamma-DR030-BarVeto000-EndVeto008"),
                             true) and
                         edm::ParameterDescription<edm::InputTag>(
                             "puppiNoLeptonsIsolationNeutralHadrons",
                             edm::InputTag("egmElectronPUPPINoLeptonsIsolation", "gamma-DR030-BarVeto000-EndVeto008"),
                             true) and
                         edm::ParameterDescription<edm::InputTag>(
                             "puppiNoLeptonsIsolationPhotons",
                             edm::InputTag("egmElectronPUPPINoLeptonsIsolation", "gamma-DR030-BarVeto000-EndVeto008"),
                             true)) or
                    false >> edm::EmptyGroupDescription());

  // embedding
  iDesc.add<bool>("embedGsfElectronCore", true)->setComment("embed external gsf electron core");
  iDesc.add<bool>("embedGsfTrack", true)->setComment("embed external gsf track");
  iDesc.add<bool>("embedSuperCluster", true)->setComment("embed external super cluster");
  iDesc.add<bool>("embedPflowSuperCluster", true)->setComment("embed external super cluster");
  iDesc.add<bool>("embedSeedCluster", true)->setComment("embed external seed cluster");
  iDesc.add<bool>("embedBasicClusters", true)->setComment("embed external basic clusters");
  iDesc.add<bool>("embedPreshowerClusters", true)->setComment("embed external preshower clusters");
  iDesc.add<bool>("embedPflowBasicClusters", true)->setComment("embed external pflow basic clusters");
  iDesc.add<bool>("embedPflowPreshowerClusters", true)->setComment("embed external pflow preshower clusters");
  iDesc.add<bool>("embedTrack", false)->setComment("embed external track");
  iDesc.add<bool>("embedRecHits", true)->setComment("embed external RecHits");

  // pf specific parameters
  iDesc.add<edm::InputTag>("pfElectronSource", edm::InputTag("pfElectrons"))
      ->setComment("particle flow input collection");
  auto&& usePfCandidateMultiMap = edm::ParameterDescription<bool>("usePfCandidateMultiMap", false, true);
  usePfCandidateMultiMap.setComment(
      "take ParticleFlow candidates from pfCandidateMultiMap instead of matching to pfElectrons by Gsf track "
      "reference");
  iDesc.ifValue(usePfCandidateMultiMap,
                true >> edm::ParameterDescription<edm::InputTag>("pfCandidateMultiMap", true) or
                    false >> edm::EmptyGroupDescription());
  iDesc.add<bool>("useParticleFlow", false)->setComment("whether to use particle flow or not");
  iDesc.add<bool>("embedPFCandidate", false)->setComment("embed external particle flow object");

  // MC matching configurables
  iDesc.add<bool>("addGenMatch", true)->setComment("add MC matching");
  iDesc.add<bool>("embedGenMatch", false)->setComment("embed MC matched MC information");
  std::vector<edm::InputTag> emptySourceVector;
  iDesc
      .addNode(edm::ParameterDescription<edm::InputTag>("genParticleMatch", edm::InputTag(), true) xor
               edm::ParameterDescription<std::vector<edm::InputTag>>("genParticleMatch", emptySourceVector, true))
      ->setComment("input with MC match information");

  // electron ID configurables
  iDesc.add<bool>("addElectronID", true)->setComment("add electron ID variables");
  edm::ParameterSetDescription electronIDSourcesPSet;
  electronIDSourcesPSet.setAllowAnything();
  iDesc
      .addNode(
          edm::ParameterDescription<edm::InputTag>("electronIDSource", edm::InputTag(), true) xor
          edm::ParameterDescription<edm::ParameterSetDescription>("electronIDSources", electronIDSourcesPSet, true))
      ->setComment("input with electron ID variables");

  // mini-iso
  iDesc.add<bool>("computeMiniIso", false)->setComment("whether or not to compute and store electron mini-isolation");
  iDesc.add<edm::InputTag>("pfCandsForMiniIso", edm::InputTag("packedPFCandidates"))
      ->setComment("collection to use to compute mini-iso");
  iDesc.add<std::vector<double>>("miniIsoParamsE", std::vector<double>())
      ->setComment("mini-iso parameters to use for endcap electrons");
  iDesc.add<std::vector<double>>("miniIsoParamsB", std::vector<double>())
      ->setComment("mini-iso parameters to use for barrel electrons");

  // IsoDeposit configurables
  edm::ParameterSetDescription isoDepositsPSet;
  isoDepositsPSet.addOptional<edm::InputTag>("tracker");
  isoDepositsPSet.addOptional<edm::InputTag>("ecal");
  isoDepositsPSet.addOptional<edm::InputTag>("hcal");
  isoDepositsPSet.addOptional<edm::InputTag>("pfAllParticles");
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
  isolationValuesPSet.addOptional<edm::InputTag>("pfAllParticles");
  isolationValuesPSet.addOptional<edm::InputTag>("pfChargedHadrons");
  isolationValuesPSet.addOptional<edm::InputTag>("pfChargedAll");
  isolationValuesPSet.addOptional<edm::InputTag>("pfPUChargedHadrons");
  isolationValuesPSet.addOptional<edm::InputTag>("pfNeutralHadrons");
  isolationValuesPSet.addOptional<edm::InputTag>("pfPhotons");
  isolationValuesPSet.addOptional<std::vector<edm::InputTag>>("user");
  iDesc.addOptional("isolationValues", isolationValuesPSet);

  // isolation values configurables
  edm::ParameterSetDescription isolationValuesNoPFIdPSet;
  isolationValuesNoPFIdPSet.addOptional<edm::InputTag>("tracker");
  isolationValuesNoPFIdPSet.addOptional<edm::InputTag>("ecal");
  isolationValuesNoPFIdPSet.addOptional<edm::InputTag>("hcal");
  isolationValuesNoPFIdPSet.addOptional<edm::InputTag>("pfAllParticles");
  isolationValuesNoPFIdPSet.addOptional<edm::InputTag>("pfChargedHadrons");
  isolationValuesNoPFIdPSet.addOptional<edm::InputTag>("pfChargedAll");
  isolationValuesNoPFIdPSet.addOptional<edm::InputTag>("pfPUChargedHadrons");
  isolationValuesNoPFIdPSet.addOptional<edm::InputTag>("pfNeutralHadrons");
  isolationValuesNoPFIdPSet.addOptional<edm::InputTag>("pfPhotons");
  isolationValuesNoPFIdPSet.addOptional<std::vector<edm::InputTag>>("user");
  iDesc.addOptional("isolationValuesNoPFId", isolationValuesNoPFIdPSet);

  // Efficiency configurables
  edm::ParameterSetDescription efficienciesPSet;
  efficienciesPSet.setAllowAnything();  // TODO: the pat helper needs to implement a description.
  iDesc.add("efficiencies", efficienciesPSet);
  iDesc.add<bool>("addEfficiencies", false);

  // Check to see if the user wants to add user data
  edm::ParameterSetDescription userDataPSet;
  PATUserDataHelper<Electron>::fillDescription(userDataPSet);
  iDesc.addOptional("userData", userDataPSet);

  // electron shapes
  iDesc.add<bool>("addMVAVariables", true)->setComment("embed extra variables in pat::Electron : sip3d, sigmaIEtaIPhi");
  iDesc.add<edm::InputTag>("reducedBarrelRecHitCollection", edm::InputTag("reducedEcalRecHitsEB"));
  iDesc.add<edm::InputTag>("reducedEndcapRecHitCollection", edm::InputTag("reducedEcalRecHitsEE"));

  edm::ParameterSetDescription isolationPSet;
  isolationPSet.setAllowAnything();  // TODO: the pat helper needs to implement a description.
  iDesc.add("userIsolation", isolationPSet);

  // Resolution configurables
  pat::helper::KinResolutionsLoader::fillDescription(iDesc);

  iDesc.add<bool>("embedHighLevelSelection", true)->setComment("embed high level selection");
  edm::ParameterSetDescription highLevelPSet;
  highLevelPSet.setAllowAnything();
  iDesc.addNode(edm::ParameterDescription<edm::InputTag>("beamLineSrc", edm::InputTag(), true))
      ->setComment("input with high level selection");
  iDesc.addNode(edm::ParameterDescription<edm::InputTag>("pvSrc", edm::InputTag(), true))
      ->setComment("input with high level selection");

  descriptions.add("PATElectronProducer", iDesc);
}

// embed various impact parameters with errors
// embed high level selection
void PATElectronProducer::embedHighLevel(pat::Electron& anElectron,
                                         reco::GsfTrackRef track,
                                         reco::TransientTrack& tt,
                                         reco::Vertex& primaryVertex,
                                         bool primaryVertexIsValid,
                                         reco::BeamSpot& beamspot,
                                         bool beamspotIsValid) {
  // Correct to PV
  // PV2D
  anElectron.setDB(track->dxy(primaryVertex.position()),
                   track->dxyError(primaryVertex.position(), primaryVertex.covariance()),
                   pat::Electron::PV2D);

  // PV3D
  std::pair<bool, Measurement1D> result =
      IPTools::signedImpactParameter3D(tt, GlobalVector(track->px(), track->py(), track->pz()), primaryVertex);
  double d0_corr = result.second.value();
  double d0_err = primaryVertexIsValid ? result.second.error() : -1.0;
  anElectron.setDB(d0_corr, d0_err, pat::Electron::PV3D);

  // Correct to beam spot
  // BS2D
  anElectron.setDB(track->dxy(beamspot), track->dxyError(beamspot), pat::Electron::BS2D);

  // make a fake vertex out of beam spot
  reco::Vertex vBeamspot(beamspot.position(), beamspot.covariance3D());

  // BS3D
  result = IPTools::signedImpactParameter3D(tt, GlobalVector(track->px(), track->py(), track->pz()), vBeamspot);
  d0_corr = result.second.value();
  d0_err = beamspotIsValid ? result.second.error() : -1.0;
  anElectron.setDB(d0_corr, d0_err, pat::Electron::BS3D);

  // PVDZ
  anElectron.setDB(
      track->dz(primaryVertex.position()), std::hypot(track->dzError(), primaryVertex.zError()), pat::Electron::PVDZ);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PATElectronProducer);

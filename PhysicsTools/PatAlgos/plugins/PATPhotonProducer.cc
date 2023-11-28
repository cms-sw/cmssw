/**
  \class    pat::PATPhotonProducer PATPhotonProducer.h "PhysicsTools/PatAlgos/interface/PATPhotonProducer.h"
  \brief    Produces the pat::Photon

   The PATPhotonProducer produces the analysis-level pat::Photon starting from
   a collection of objects of PhotonType.

  \author   Steven Lowette
  \version  $Id: PATPhotonProducer.h,v 1.19 2009/06/25 23:49:35 gpetrucc Exp $
*/

#include "CommonTools/Egamma/interface/ConversionTools.h"
#include "CommonTools/Utils/interface/EtComparator.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/UserData.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/EmptyGroupDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/CaloTopologyRecord.h"
#include "PhysicsTools/PatAlgos/interface/EfficiencyLoader.h"
#include "PhysicsTools/PatAlgos/interface/KinResolutionsLoader.h"
#include "PhysicsTools/PatAlgos/interface/MultiIsolator.h"
#include "PhysicsTools/PatAlgos/interface/PATUserDataHelper.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"
#include "RecoEgamma/EgammaTools/interface/EcalRegressionData.h"

namespace pat {

  class PATPhotonProducer : public edm::stream::EDProducer<> {
  public:
    explicit PATPhotonProducer(const edm::ParameterSet& iConfig);
    ~PATPhotonProducer() override;

    void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    // configurables
    edm::EDGetTokenT<edm::View<reco::Photon>> photonToken_;
    edm::EDGetTokenT<reco::GsfElectronCollection> electronToken_;
    edm::EDGetTokenT<reco::ConversionCollection> hConversionsToken_;
    edm::EDGetTokenT<reco::BeamSpot> beamLineToken_;

    bool embedSuperCluster_;
    bool embedSeedCluster_;
    bool embedBasicClusters_;
    bool embedPreshowerClusters_;
    bool embedRecHits_;

    edm::InputTag reducedBarrelRecHitCollection_;
    edm::EDGetTokenT<EcalRecHitCollection> reducedBarrelRecHitCollectionToken_;
    edm::InputTag reducedEndcapRecHitCollection_;
    edm::EDGetTokenT<EcalRecHitCollection> reducedEndcapRecHitCollectionToken_;

    const EcalClusterLazyTools::ESGetTokens ecalClusterToolsESGetTokens_;

    bool addPFClusterIso_;
    bool addPuppiIsolation_;
    edm::EDGetTokenT<edm::ValueMap<float>> ecalPFClusterIsoT_;
    edm::EDGetTokenT<edm::ValueMap<float>> hcalPFClusterIsoT_;

    bool addGenMatch_;
    bool embedGenMatch_;
    std::vector<edm::EDGetTokenT<edm::Association<reco::GenParticleCollection>>> genMatchTokens_;

    // tools
    GreaterByEt<Photon> eTComparator_;

    typedef std::vector<edm::Handle<edm::ValueMap<IsoDeposit>>> IsoDepositMaps;
    typedef std::vector<edm::Handle<edm::ValueMap<double>>> IsolationValueMaps;
    typedef std::pair<pat::IsolationKeys, edm::InputTag> IsolationLabel;
    typedef std::vector<IsolationLabel> IsolationLabels;

    pat::helper::MultiIsolator isolator_;
    pat::helper::MultiIsolator::IsolationValuePairs isolatorTmpStorage_;  // better here than recreate at each event
    std::vector<edm::EDGetTokenT<edm::ValueMap<IsoDeposit>>> isoDepositTokens_;
    std::vector<edm::EDGetTokenT<edm::ValueMap<double>>> isolationValueTokens_;

    IsolationLabels isoDepositLabels_;
    IsolationLabels isolationValueLabels_;

    /// fill the labels vector from the contents of the parameter set,
    /// for the isodeposit or isolation values embedding
    template <typename T>
    void readIsolationLabels(const edm::ParameterSet& iConfig,
                             const char* psetName,
                             IsolationLabels& labels,
                             std::vector<edm::EDGetTokenT<edm::ValueMap<T>>>& tokens);

    bool addEfficiencies_;
    pat::helper::EfficiencyLoader efficiencyLoader_;

    bool addResolutions_;
    pat::helper::KinResolutionsLoader resolutionLoader_;

    bool addPhotonID_;
    typedef std::pair<std::string, edm::InputTag> NameTag;
    std::vector<NameTag> photIDSrcs_;
    std::vector<edm::EDGetTokenT<edm::ValueMap<Bool_t>>> photIDTokens_;

    bool useUserData_;
    //PUPPI isolation tokens
    edm::EDGetTokenT<edm::ValueMap<float>> PUPPIIsolation_charged_hadrons_;
    edm::EDGetTokenT<edm::ValueMap<float>> PUPPIIsolation_neutral_hadrons_;
    edm::EDGetTokenT<edm::ValueMap<float>> PUPPIIsolation_photons_;
    pat::PATUserDataHelper<pat::Photon> userDataHelper_;

    const CaloTopology* ecalTopology_;
    const CaloGeometry* ecalGeometry_;

    bool saveRegressionData_;

    const edm::ESGetToken<CaloTopology, CaloTopologyRecord> ecalTopologyToken_;
    const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> ecalGeometryToken_;
  };

}  // namespace pat

template <typename T>
void pat::PATPhotonProducer::readIsolationLabels(const edm::ParameterSet& iConfig,
                                                 const char* psetName,
                                                 pat::PATPhotonProducer::IsolationLabels& labels,
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

PATPhotonProducer::PATPhotonProducer(const edm::ParameterSet& iConfig)
    : ecalClusterToolsESGetTokens_{consumesCollector()},
      isolator_(iConfig.getParameter<edm::ParameterSet>("userIsolation"), consumesCollector(), false),
      useUserData_(iConfig.exists("userData")),
      ecalTopologyToken_{esConsumes()},
      ecalGeometryToken_{esConsumes()} {
  // initialize the configurables
  photonToken_ = consumes<edm::View<reco::Photon>>(iConfig.getParameter<edm::InputTag>("photonSource"));
  electronToken_ = consumes<reco::GsfElectronCollection>(iConfig.getParameter<edm::InputTag>("electronSource"));
  hConversionsToken_ = consumes<reco::ConversionCollection>(iConfig.getParameter<edm::InputTag>("conversionSource"));
  beamLineToken_ = consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamLineSrc"));
  embedSuperCluster_ = iConfig.getParameter<bool>("embedSuperCluster");
  embedSeedCluster_ = iConfig.getParameter<bool>("embedSeedCluster");
  embedBasicClusters_ = iConfig.getParameter<bool>("embedBasicClusters");
  embedPreshowerClusters_ = iConfig.getParameter<bool>("embedPreshowerClusters");
  embedRecHits_ = iConfig.getParameter<bool>("embedRecHits");
  reducedBarrelRecHitCollection_ = iConfig.getParameter<edm::InputTag>("reducedBarrelRecHitCollection");
  reducedBarrelRecHitCollectionToken_ = mayConsume<EcalRecHitCollection>(reducedBarrelRecHitCollection_);
  reducedEndcapRecHitCollection_ = iConfig.getParameter<edm::InputTag>("reducedEndcapRecHitCollection");
  reducedEndcapRecHitCollectionToken_ = mayConsume<EcalRecHitCollection>(reducedEndcapRecHitCollection_);
  // MC matching configurables
  addGenMatch_ = iConfig.getParameter<bool>("addGenMatch");
  if (addGenMatch_) {
    embedGenMatch_ = iConfig.getParameter<bool>("embedGenMatch");
    genMatchTokens_.push_back(consumes<edm::Association<reco::GenParticleCollection>>(
        iConfig.getParameter<edm::InputTag>("genParticleMatch")));
  }
  // Efficiency configurables
  addEfficiencies_ = iConfig.getParameter<bool>("addEfficiencies");
  if (addEfficiencies_) {
    efficiencyLoader_ =
        pat::helper::EfficiencyLoader(iConfig.getParameter<edm::ParameterSet>("efficiencies"), consumesCollector());
  }
  // PFCluster Isolation maps
  addPuppiIsolation_ = iConfig.getParameter<bool>("addPuppiIsolation");
  if (addPuppiIsolation_) {
    PUPPIIsolation_charged_hadrons_ =
        consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("puppiIsolationChargedHadrons"));
    PUPPIIsolation_neutral_hadrons_ =
        consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("puppiIsolationNeutralHadrons"));
    PUPPIIsolation_photons_ =
        consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("puppiIsolationPhotons"));
  }
  addPFClusterIso_ = iConfig.getParameter<bool>("addPFClusterIso");
  if (addPFClusterIso_) {
    ecalPFClusterIsoT_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("ecalPFClusterIsoMap"));
    auto hcPFC = iConfig.getParameter<edm::InputTag>("hcalPFClusterIsoMap");
    if (not hcPFC.label().empty())
      hcalPFClusterIsoT_ = consumes<edm::ValueMap<float>>(hcPFC);
  }

  // photon ID configurables
  addPhotonID_ = iConfig.getParameter<bool>("addPhotonID");
  if (addPhotonID_) {
    // it might be a single photon ID
    if (iConfig.existsAs<edm::InputTag>("photonIDSource")) {
      photIDSrcs_.push_back(NameTag("", iConfig.getParameter<edm::InputTag>("photonIDSource")));
    }
    // or there might be many of them
    if (iConfig.existsAs<edm::ParameterSet>("photonIDSources")) {
      // please don't configure me twice
      if (!photIDSrcs_.empty()) {
        throw cms::Exception("Configuration")
            << "PATPhotonProducer: you can't specify both 'photonIDSource' and 'photonIDSources'\n";
      }
      // read the different photon ID names
      edm::ParameterSet idps = iConfig.getParameter<edm::ParameterSet>("photonIDSources");
      std::vector<std::string> names = idps.getParameterNamesForType<edm::InputTag>();
      for (std::vector<std::string>::const_iterator it = names.begin(), ed = names.end(); it != ed; ++it) {
        photIDSrcs_.push_back(NameTag(*it, idps.getParameter<edm::InputTag>(*it)));
      }
    }
    // but in any case at least once
    if (photIDSrcs_.empty())
      throw cms::Exception("Configuration") << "PATPhotonProducer: id addPhotonID is true, you must specify either:\n"
                                            << "\tInputTag photonIDSource = <someTag>\n"
                                            << "or\n"
                                            << "\tPSet photonIDSources = { \n"
                                            << "\t\tInputTag <someName> = <someTag>   // as many as you want \n "
                                            << "\t}\n";
  }
  photIDTokens_ = edm::vector_transform(
      photIDSrcs_, [this](NameTag const& tag) { return mayConsume<edm::ValueMap<Bool_t>>(tag.second); });
  // Resolution configurables
  addResolutions_ = iConfig.getParameter<bool>("addResolutions");
  if (addResolutions_) {
    resolutionLoader_ =
        pat::helper::KinResolutionsLoader(iConfig.getParameter<edm::ParameterSet>("resolutions"), consumesCollector());
  }
  // Check to see if the user wants to add user data
  if (useUserData_) {
    userDataHelper_ =
        PATUserDataHelper<Photon>(iConfig.getParameter<edm::ParameterSet>("userData"), consumesCollector());
  }
  // produces vector of photons
  produces<std::vector<Photon>>();

  // read isoDeposit labels, for direct embedding
  readIsolationLabels(iConfig, "isoDeposits", isoDepositLabels_, isoDepositTokens_);
  // read isolation value labels, for direct embedding
  readIsolationLabels(iConfig, "isolationValues", isolationValueLabels_, isolationValueTokens_);

  saveRegressionData_ = iConfig.getParameter<bool>("saveRegressionData");
}

PATPhotonProducer::~PATPhotonProducer() {}

void PATPhotonProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // switch off embedding (in unschedules mode)
  if (iEvent.isRealData()) {
    addGenMatch_ = false;
    embedGenMatch_ = false;
  }

  ecalTopology_ = &iSetup.getData(ecalTopologyToken_);
  ecalGeometry_ = &iSetup.getData(ecalGeometryToken_);

  // Get the vector of Photon's from the event
  edm::Handle<edm::View<reco::Photon>> photons;
  iEvent.getByToken(photonToken_, photons);

  // for conversion veto selection
  edm::Handle<reco::ConversionCollection> hConversions;
  iEvent.getByToken(hConversionsToken_, hConversions);

  // Get the collection of electrons from the event
  edm::Handle<reco::GsfElectronCollection> hElectrons;
  iEvent.getByToken(electronToken_, hElectrons);

  // Get the beamspot
  edm::Handle<reco::BeamSpot> beamSpotHandle;
  iEvent.getByToken(beamLineToken_, beamSpotHandle);

  EcalClusterLazyTools lazyTools(iEvent,
                                 ecalClusterToolsESGetTokens_.get(iSetup),
                                 reducedBarrelRecHitCollectionToken_,
                                 reducedEndcapRecHitCollectionToken_);

  // prepare the MC matching
  std::vector<edm::Handle<edm::Association<reco::GenParticleCollection>>> genMatches(genMatchTokens_.size());
  if (addGenMatch_) {
    for (size_t j = 0, nd = genMatchTokens_.size(); j < nd; ++j) {
      iEvent.getByToken(genMatchTokens_[j], genMatches[j]);
    }
  }

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

  // prepare ID extraction
  std::vector<edm::Handle<edm::ValueMap<Bool_t>>> idhandles;
  std::vector<pat::Photon::IdPair> ids;
  if (addPhotonID_) {
    idhandles.resize(photIDSrcs_.size());
    ids.resize(photIDSrcs_.size());
    for (size_t i = 0; i < photIDSrcs_.size(); ++i) {
      iEvent.getByToken(photIDTokens_[i], idhandles[i]);
      ids[i].first = photIDSrcs_[i].first;
    }
  }

  //value maps for puppi isolation
  edm::Handle<edm::ValueMap<float>> PUPPIIsolation_charged_hadrons;
  edm::Handle<edm::ValueMap<float>> PUPPIIsolation_neutral_hadrons;
  edm::Handle<edm::ValueMap<float>> PUPPIIsolation_photons;
  if (addPuppiIsolation_) {
    iEvent.getByToken(PUPPIIsolation_charged_hadrons_, PUPPIIsolation_charged_hadrons);
    iEvent.getByToken(PUPPIIsolation_neutral_hadrons_, PUPPIIsolation_neutral_hadrons);
    iEvent.getByToken(PUPPIIsolation_photons_, PUPPIIsolation_photons);
  }

  // loop over photons
  std::vector<Photon>* PATPhotons = new std::vector<Photon>();
  for (edm::View<reco::Photon>::const_iterator itPhoton = photons->begin(); itPhoton != photons->end(); itPhoton++) {
    // construct the Photon from the ref -> save ref to original object
    unsigned int idx = itPhoton - photons->begin();
    edm::RefToBase<reco::Photon> photonRef = photons->refAt(idx);
    edm::Ptr<reco::Photon> photonPtr = photons->ptrAt(idx);
    Photon aPhoton(photonRef);
    auto phoPtr = photons->ptrAt(idx);
    if (embedSuperCluster_)
      aPhoton.embedSuperCluster();
    if (embedSeedCluster_)
      aPhoton.embedSeedCluster();
    if (embedBasicClusters_)
      aPhoton.embedBasicClusters();
    if (embedPreshowerClusters_)
      aPhoton.embedPreshowerClusters();

    std::vector<DetId> selectedCells;
    bool barrel = itPhoton->isEB();
    //loop over sub clusters
    if (embedBasicClusters_) {
      for (reco::CaloCluster_iterator clusIt = itPhoton->superCluster()->clustersBegin();
           clusIt != itPhoton->superCluster()->clustersEnd();
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

    edm::Handle<EcalRecHitCollection> recHitsEBHandle;
    iEvent.getByToken(reducedBarrelRecHitCollectionToken_, recHitsEBHandle);
    edm::Handle<EcalRecHitCollection> recHitsEEHandle;
    iEvent.getByToken(reducedEndcapRecHitCollectionToken_, recHitsEEHandle);

    //orginal code would throw an exception via the handle not being valid but now it'll just have a null pointer error
    //should have little effect, if its not barrel or endcap, something very bad has happened elsewhere anyways
    const EcalRecHitCollection* recHits = nullptr;
    if (photonRef->superCluster()->seed()->hitsAndFractions().at(0).first.subdetId() == EcalBarrel)
      recHits = recHitsEBHandle.product();
    else if (photonRef->superCluster()->seed()->hitsAndFractions().at(0).first.subdetId() == EcalEndcap)
      recHits = recHitsEEHandle.product();

    EcalRecHitCollection selectedRecHits;

    unsigned nSelectedCells = selectedCells.size();
    for (unsigned icell = 0; icell < nSelectedCells; ++icell) {
      EcalRecHitCollection::const_iterator it = recHits->find(selectedCells[icell]);
      if (it != recHits->end()) {
        selectedRecHits.push_back(*it);
      }
    }
    selectedRecHits.sort();
    if (embedRecHits_)
      aPhoton.embedRecHits(&selectedRecHits);

    // store the match to the generated final state muons
    if (addGenMatch_) {
      for (size_t i = 0, n = genMatches.size(); i < n; ++i) {
        reco::GenParticleRef genPhoton = (*genMatches[i])[photonRef];
        aPhoton.addGenParticleRef(genPhoton);
      }
      if (embedGenMatch_)
        aPhoton.embedGenParticle();
    }

    if (efficiencyLoader_.enabled()) {
      efficiencyLoader_.setEfficiencies(aPhoton, photonRef);
    }

    if (resolutionLoader_.enabled()) {
      resolutionLoader_.setResolutions(aPhoton);
    }

    // here comes the extra functionality
    if (isolator_.enabled()) {
      isolator_.fill(*photons, idx, isolatorTmpStorage_);
      typedef pat::helper::MultiIsolator::IsolationValuePairs IsolationValuePairs;
      // better to loop backwards, so the vector is resized less times
      for (IsolationValuePairs::const_reverse_iterator it = isolatorTmpStorage_.rbegin(),
                                                       ed = isolatorTmpStorage_.rend();
           it != ed;
           ++it) {
        aPhoton.setIsolation(it->first, it->second);
      }
    }

    for (size_t j = 0, nd = deposits.size(); j < nd; ++j) {
      aPhoton.setIsoDeposit(isoDepositLabels_[j].first, (*deposits[j])[photonRef]);
    }

    for (size_t j = 0; j < isolationValues.size(); ++j) {
      aPhoton.setIsolation(isolationValueLabels_[j].first, (*isolationValues[j])[photonRef]);
    }

    // add photon ID info
    if (addPhotonID_) {
      for (size_t i = 0; i < photIDSrcs_.size(); ++i) {
        ids[i].second = (*idhandles[i])[photonRef];
      }
      aPhoton.setPhotonIDs(ids);
    }

    if (useUserData_) {
      userDataHelper_.add(aPhoton, iEvent, iSetup);
    }

    // set conversion veto selection
    bool passelectronveto = false;
    if (hConversions.isValid()) {
      // this is recommended method
      passelectronveto = !ConversionTools::hasMatchedPromptElectron(
          photonRef->superCluster(), *hElectrons, *hConversions, beamSpotHandle->position());
    }
    aPhoton.setPassElectronVeto(passelectronveto);

    // set electron veto using pixel seed (not recommended but many analysis groups are still using since it is powerful method to remove electrons)
    aPhoton.setHasPixelSeed(photonRef->hasPixelSeed());

    // set seed energy
    aPhoton.setSeedEnergy(photonRef->superCluster()->seed()->energy());

    // set input variables for regression energy correction
    if (saveRegressionData_) {
      EcalRegressionData ecalRegData;
      ecalRegData.fill(*(photonRef->superCluster()),
                       recHitsEBHandle.product(),
                       recHitsEEHandle.product(),
                       ecalGeometry_,
                       ecalTopology_,
                       -1);

      aPhoton.setEMax(ecalRegData.eMax());
      aPhoton.setE2nd(ecalRegData.e2nd());
      aPhoton.setE3x3(ecalRegData.e3x3());
      aPhoton.setETop(ecalRegData.eTop());
      aPhoton.setEBottom(ecalRegData.eBottom());
      aPhoton.setELeft(ecalRegData.eLeft());
      aPhoton.setERight(ecalRegData.eRight());
      aPhoton.setSee(ecalRegData.sigmaIEtaIEta());
      aPhoton.setSep(
          ecalRegData.sigmaIEtaIPhi() * ecalRegData.sigmaIEtaIEta() *
          ecalRegData
              .sigmaIPhiIPhi());  //there is a conflict on what sigmaIEtaIPhi actually is, regression and ID have it differently, this may change in later releases
      aPhoton.setSpp(ecalRegData.sigmaIPhiIPhi());

      aPhoton.setMaxDR(ecalRegData.maxSubClusDR());
      aPhoton.setMaxDRDPhi(ecalRegData.maxSubClusDRDPhi());
      aPhoton.setMaxDRDEta(ecalRegData.maxSubClusDRDEta());
      aPhoton.setMaxDRRawEnergy(ecalRegData.maxSubClusDRRawEnergy());
      aPhoton.setSubClusRawE1(ecalRegData.subClusRawEnergy(EcalRegressionData::SubClusNr::C1));
      aPhoton.setSubClusRawE2(ecalRegData.subClusRawEnergy(EcalRegressionData::SubClusNr::C2));
      aPhoton.setSubClusRawE3(ecalRegData.subClusRawEnergy(EcalRegressionData::SubClusNr::C3));
      aPhoton.setSubClusDPhi1(ecalRegData.subClusDPhi(EcalRegressionData::SubClusNr::C1));
      aPhoton.setSubClusDPhi2(ecalRegData.subClusDPhi(EcalRegressionData::SubClusNr::C2));
      aPhoton.setSubClusDPhi3(ecalRegData.subClusDPhi(EcalRegressionData::SubClusNr::C3));
      aPhoton.setSubClusDEta1(ecalRegData.subClusDEta(EcalRegressionData::SubClusNr::C1));
      aPhoton.setSubClusDEta2(ecalRegData.subClusDEta(EcalRegressionData::SubClusNr::C2));
      aPhoton.setSubClusDEta3(ecalRegData.subClusDEta(EcalRegressionData::SubClusNr::C3));

      aPhoton.setCryPhi(ecalRegData.seedCrysPhiOrY());
      aPhoton.setCryEta(ecalRegData.seedCrysEtaOrX());
      aPhoton.setIEta(ecalRegData.seedCrysIEtaOrIX());
      aPhoton.setIPhi(ecalRegData.seedCrysIPhiOrIY());
    } else {
      aPhoton.setEMax(0);
      aPhoton.setE2nd(0);
      aPhoton.setE3x3(0);
      aPhoton.setETop(0);
      aPhoton.setEBottom(0);
      aPhoton.setELeft(0);
      aPhoton.setERight(0);
      aPhoton.setSee(0);
      aPhoton.setSep(0);
      aPhoton.setSpp(0);

      aPhoton.setMaxDR(0);
      aPhoton.setMaxDRDPhi(0);
      aPhoton.setMaxDRDEta(0);
      aPhoton.setMaxDRRawEnergy(0);
      aPhoton.setSubClusRawE1(0);
      aPhoton.setSubClusRawE2(0);
      aPhoton.setSubClusRawE3(0);
      aPhoton.setSubClusDPhi1(0);
      aPhoton.setSubClusDPhi2(0);
      aPhoton.setSubClusDPhi3(0);
      aPhoton.setSubClusDEta1(0);
      aPhoton.setSubClusDEta2(0);
      aPhoton.setSubClusDEta3(0);

      aPhoton.setCryPhi(0);
      aPhoton.setCryEta(0);
      aPhoton.setIEta(0);
      aPhoton.setIPhi(0);
    }

    if (addPuppiIsolation_)
      aPhoton.setIsolationPUPPI((*PUPPIIsolation_charged_hadrons)[phoPtr],
                                (*PUPPIIsolation_neutral_hadrons)[phoPtr],
                                (*PUPPIIsolation_photons)[phoPtr]);
    else
      aPhoton.setIsolationPUPPI(-999., -999., -999.);

    // Get PFCluster Isolation
    if (addPFClusterIso_) {
      reco::Photon::PflowIsolationVariables newPFIsol = aPhoton.getPflowIsolationVariables();
      edm::Handle<edm::ValueMap<float>> ecalPFClusterIsoMapH;
      iEvent.getByToken(ecalPFClusterIsoT_, ecalPFClusterIsoMapH);
      newPFIsol.sumEcalClusterEt = (*ecalPFClusterIsoMapH)[photonRef];
      edm::Handle<edm::ValueMap<float>> hcalPFClusterIsoMapH;
      if (not hcalPFClusterIsoT_.isUninitialized()) {
        iEvent.getByToken(hcalPFClusterIsoT_, hcalPFClusterIsoMapH);
        newPFIsol.sumHcalClusterEt = (*hcalPFClusterIsoMapH)[photonRef];
      } else {
        newPFIsol.sumHcalClusterEt = -999.;
      }
      aPhoton.setPflowIsolationVariables(newPFIsol);
    }

    // add the Photon to the vector of Photons
    PATPhotons->push_back(aPhoton);
  }

  // sort Photons in ET
  std::sort(PATPhotons->begin(), PATPhotons->end(), eTComparator_);

  // put genEvt object in Event
  std::unique_ptr<std::vector<Photon>> myPhotons(PATPhotons);
  iEvent.put(std::move(myPhotons));
  if (isolator_.enabled())
    isolator_.endEvent();
}

// ParameterSet description for module
void PATPhotonProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription iDesc;
  iDesc.setComment("PAT photon producer module");

  // input source
  iDesc.add<edm::InputTag>("photonSource", edm::InputTag("no default"))->setComment("input collection");
  iDesc.add<edm::InputTag>("electronSource", edm::InputTag("no default"))->setComment("input collection");
  iDesc.add<edm::InputTag>("conversionSource", edm::InputTag("allConversions"))->setComment("input collection");

  iDesc.add<edm::InputTag>("reducedBarrelRecHitCollection", edm::InputTag("reducedEcalRecHitsEB"));
  iDesc.add<edm::InputTag>("reducedEndcapRecHitCollection", edm::InputTag("reducedEcalRecHitsEE"));

  iDesc.ifValue(
      edm::ParameterDescription<bool>("addPFClusterIso", false, true),
      true >> (edm::ParameterDescription<edm::InputTag>(
                   "ecalPFClusterIsoMap", edm::InputTag("photonEcalPFClusterIsolationProducer"), true) and
               edm::ParameterDescription<edm::InputTag>(
                   "hcalPFClusterIsoMap", edm::InputTag("photonHcalPFClusterIsolationProducer"), true)) or
          false >> (edm::ParameterDescription<edm::InputTag>("ecalPFClusterIsoMap", edm::InputTag(""), true) and
                    edm::ParameterDescription<edm::InputTag>("hcalPFClusterIsoMap", edm::InputTag(""), true)));

  iDesc.ifValue(
      edm::ParameterDescription<bool>("addPuppiIsolation", false, true),
      true >> (edm::ParameterDescription<edm::InputTag>(
                   "puppiIsolationChargedHadrons", edm::InputTag("egmPhotonPUPPIIsolation", "h+-DR030-"), true) and
               edm::ParameterDescription<edm::InputTag>(
                   "puppiIsolationNeutralHadrons", edm::InputTag("egmPhotonPUPPIIsolation", "h0-DR030-"), true) and
               edm::ParameterDescription<edm::InputTag>(
                   "puppiIsolationPhotons", edm::InputTag("egmPhotonPUPPIIsolation", "gamma-DR030-"), true)) or
          false >> edm::EmptyGroupDescription());

  iDesc.add<bool>("embedSuperCluster", true)->setComment("embed external super cluster");
  iDesc.add<bool>("embedSeedCluster", true)->setComment("embed external seed cluster");
  iDesc.add<bool>("embedBasicClusters", true)->setComment("embed external basic clusters");
  iDesc.add<bool>("embedPreshowerClusters", true)->setComment("embed external preshower clusters");
  iDesc.add<bool>("embedRecHits", true)->setComment("embed external RecHits");

  // MC matching configurables
  iDesc.add<bool>("addGenMatch", true)->setComment("add MC matching");
  iDesc.add<bool>("embedGenMatch", false)->setComment("embed MC matched MC information");
  std::vector<edm::InputTag> emptySourceVector;
  iDesc
      .addNode(edm::ParameterDescription<edm::InputTag>("genParticleMatch", edm::InputTag(), true) xor
               edm::ParameterDescription<std::vector<edm::InputTag>>("genParticleMatch", emptySourceVector, true))
      ->setComment("input with MC match information");

  pat::helper::KinResolutionsLoader::fillDescription(iDesc);

  // photon ID configurables
  iDesc.add<bool>("addPhotonID", true)->setComment("add photon ID variables");
  edm::ParameterSetDescription photonIDSourcesPSet;
  photonIDSourcesPSet.setAllowAnything();
  iDesc
      .addNode(edm::ParameterDescription<edm::InputTag>("photonIDSource", edm::InputTag(), true) xor
               edm::ParameterDescription<edm::ParameterSetDescription>("photonIDSources", photonIDSourcesPSet, true))
      ->setComment("input with photon ID variables");

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

  // Efficiency configurables
  edm::ParameterSetDescription efficienciesPSet;
  efficienciesPSet.setAllowAnything();  // TODO: the pat helper needs to implement a description.
  iDesc.add("efficiencies", efficienciesPSet);
  iDesc.add<bool>("addEfficiencies", false);

  // Check to see if the user wants to add user data
  edm::ParameterSetDescription userDataPSet;
  PATUserDataHelper<Photon>::fillDescription(userDataPSet);
  iDesc.addOptional("userData", userDataPSet);

  edm::ParameterSetDescription isolationPSet;
  isolationPSet.setAllowAnything();  // TODO: the pat helper needs to implement a description.
  iDesc.add("userIsolation", isolationPSet);

  iDesc.addNode(edm::ParameterDescription<edm::InputTag>("beamLineSrc", edm::InputTag(), true))
      ->setComment("input with high level selection");

  iDesc.add<bool>("saveRegressionData", true)->setComment("save regression input variables");

  descriptions.add("PATPhotonProducer", iDesc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PATPhotonProducer);

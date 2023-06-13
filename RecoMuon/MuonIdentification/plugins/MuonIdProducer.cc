// -*- C++ -*-
//
// Package:    MuonIdentification
// Class:      MuonIdProducer
//
//
// Original Author:  Dmytro Kovalskyi
//
//

// user include files
#include "RecoMuon/MuonIdentification/plugins/MuonIdProducer.h"

#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/MuonDetId/interface/ME0DetId.h"
#include "DataFormats/MuonReco/interface/MuonCocktails.h"
#include "DataFormats/MuonReco/interface/MuonTime.h"
#include "DataFormats/MuonReco/interface/MuonTimeExtra.h"
#include "DataFormats/MuonReco/interface/MuonTimeExtraMap.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositFwd.h"

#include "PhysicsTools/IsolationAlgos/interface/IsoDepositExtractorFactory.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "RecoMuon/MuonIdentification/interface/MuonMesh.h"
#include "RecoMuon/MuonIdentification/interface/MuonKinkFinder.h"

MuonIdProducer::MuonIdProducer(const edm::ParameterSet& iConfig)
    : geomTokenRun_(esConsumes<edm::Transition::BeginRun>()),
      propagatorToken_(esConsumes(edm::ESInputTag("", "SteppingHelixPropagatorAny"))) {
  LogTrace("MuonIdentification") << "RecoMuon/MuonIdProducer :: Constructor called";

  produces<reco::MuonCollection>();
  produces<reco::CaloMuonCollection>();
  produces<reco::MuonTimeExtraMap>("combined");
  produces<reco::MuonTimeExtraMap>("dt");
  produces<reco::MuonTimeExtraMap>("csc");

  minPt_ = iConfig.getParameter<double>("minPt");
  minP_ = iConfig.getParameter<double>("minP");
  minPCaloMuon_ = iConfig.getParameter<double>("minPCaloMuon");
  minNumberOfMatches_ = iConfig.getParameter<int>("minNumberOfMatches");
  addExtraSoftMuons_ = iConfig.getParameter<bool>("addExtraSoftMuons");
  maxAbsEta_ = iConfig.getParameter<double>("maxAbsEta");
  maxAbsDx_ = iConfig.getParameter<double>("maxAbsDx");
  maxAbsPullX2_ = iConfig.getParameter<double>("maxAbsPullX");
  maxAbsPullX2_ *= maxAbsPullX2_;
  maxAbsDy_ = iConfig.getParameter<double>("maxAbsDy");
  maxAbsPullY2_ = iConfig.getParameter<double>("maxAbsPullY");
  maxAbsPullY2_ *= maxAbsPullY2_;
  fillCaloCompatibility_ = iConfig.getParameter<bool>("fillCaloCompatibility");
  fillEnergy_ = iConfig.getParameter<bool>("fillEnergy");
  storeCrossedHcalRecHits_ = iConfig.getParameter<bool>("storeCrossedHcalRecHits");
  fillMatching_ = iConfig.getParameter<bool>("fillMatching");
  fillIsolation_ = iConfig.getParameter<bool>("fillIsolation");
  fillShowerDigis_ = iConfig.getParameter<bool>("fillShowerDigis");
  writeIsoDeposits_ = iConfig.getParameter<bool>("writeIsoDeposits");
  fillGlobalTrackQuality_ = iConfig.getParameter<bool>("fillGlobalTrackQuality");
  fillGlobalTrackRefits_ = iConfig.getParameter<bool>("fillGlobalTrackRefits");
  arbitrateTrackerMuons_ = iConfig.getParameter<bool>("arbitrateTrackerMuons");
  selectHighPurity_ = iConfig.getParameter<bool>("selectHighPurity");
  //SK: (maybe temporary) run it only if the global is also run
  fillTrackerKink_ = false;
  if (fillGlobalTrackQuality_)
    fillTrackerKink_ = iConfig.getParameter<bool>("fillTrackerKink");

  ptThresholdToFillCandidateP4WithGlobalFit_ =
      iConfig.getParameter<double>("ptThresholdToFillCandidateP4WithGlobalFit");
  sigmaThresholdToFillCandidateP4WithGlobalFit_ =
      iConfig.getParameter<double>("sigmaThresholdToFillCandidateP4WithGlobalFit");
  caloCut_ = iConfig.getParameter<double>("minCaloCompatibility");  //CaloMuons
  arbClean_ = iConfig.getParameter<bool>("runArbitrationCleaner");  // muon mesh

  // Load TrackDetectorAssociator parameters
  const edm::ParameterSet parameters = iConfig.getParameter<edm::ParameterSet>("TrackAssociatorParameters");
  edm::ConsumesCollector iC = consumesCollector();
  parameters_.loadParameters(parameters, iC);

  // Load parameters for the TimingFiller
  edm::ParameterSet timingParameters = iConfig.getParameter<edm::ParameterSet>("TimingFillerParameters");
  theTimingFiller_ = std::make_unique<MuonTimingFiller>(timingParameters, consumesCollector());

  // Load parameters for the ShowerDigiFiller
  if (fillShowerDigis_ && fillMatching_) {
    edm::ParameterSet showerDigiParameters = iConfig.getParameter<edm::ParameterSet>("ShowerDigiFillerParameters");
    theShowerDigiFiller_ = std::make_unique<MuonShowerDigiFiller>(showerDigiParameters, consumesCollector());
  } else {
    theShowerDigiFiller_ = std::make_unique<MuonShowerDigiFiller>();  // to be used to call fillDefault only
  }

  if (fillCaloCompatibility_) {
    // Load MuonCaloCompatibility parameters
    const auto caloParams = iConfig.getParameter<edm::ParameterSet>("MuonCaloCompatibility");
    muonCaloCompatibility_.configure(caloParams);
  }

  if (fillIsolation_) {
    // Load MuIsoExtractor parameters
    edm::ParameterSet caloExtractorPSet = iConfig.getParameter<edm::ParameterSet>("CaloExtractorPSet");
    std::string caloExtractorName = caloExtractorPSet.getParameter<std::string>("ComponentName");
    muIsoExtractorCalo_ =
        IsoDepositExtractorFactory::get()->create(caloExtractorName, caloExtractorPSet, consumesCollector());

    edm::ParameterSet trackExtractorPSet = iConfig.getParameter<edm::ParameterSet>("TrackExtractorPSet");
    std::string trackExtractorName = trackExtractorPSet.getParameter<std::string>("ComponentName");
    muIsoExtractorTrack_ =
        IsoDepositExtractorFactory::get()->create(trackExtractorName, trackExtractorPSet, consumesCollector());

    edm::ParameterSet jetExtractorPSet = iConfig.getParameter<edm::ParameterSet>("JetExtractorPSet");
    std::string jetExtractorName = jetExtractorPSet.getParameter<std::string>("ComponentName");
    muIsoExtractorJet_ =
        IsoDepositExtractorFactory::get()->create(jetExtractorName, jetExtractorPSet, consumesCollector());
  }
  if (fillIsolation_ && writeIsoDeposits_) {
    trackDepositName_ = iConfig.getParameter<std::string>("trackDepositName");
    produces<reco::IsoDepositMap>(trackDepositName_);
    ecalDepositName_ = iConfig.getParameter<std::string>("ecalDepositName");
    produces<reco::IsoDepositMap>(ecalDepositName_);
    hcalDepositName_ = iConfig.getParameter<std::string>("hcalDepositName");
    produces<reco::IsoDepositMap>(hcalDepositName_);
    hoDepositName_ = iConfig.getParameter<std::string>("hoDepositName");
    produces<reco::IsoDepositMap>(hoDepositName_);
    jetDepositName_ = iConfig.getParameter<std::string>("jetDepositName");
    produces<reco::IsoDepositMap>(jetDepositName_);
  }

  inputCollectionLabels_ = iConfig.getParameter<std::vector<edm::InputTag> >("inputCollectionLabels");
  const auto inputCollectionTypes = iConfig.getParameter<std::vector<std::string> >("inputCollectionTypes");
  if (inputCollectionLabels_.size() != inputCollectionTypes.size())
    throw cms::Exception("ConfigurationError")
        << "Number of input collection labels is different from number of types. "
        << "For each collection label there should be exactly one collection type specified.";
  if (inputCollectionLabels_.size() > 7 || inputCollectionLabels_.empty())
    throw cms::Exception("ConfigurationError") << "Number of input collections should be from 1 to 7.";

  debugWithTruthMatching_ = iConfig.getParameter<bool>("debugWithTruthMatching");
  if (debugWithTruthMatching_) {
    edm::LogWarning("MuonIdentification")
        << "========================================================================\n"
        << "Debugging mode with truth matching is turned on!!! Make sure you understand what you are doing!\n"
        << "========================================================================\n";

    globalGeomToken_ = esConsumes();
  }
  if (fillGlobalTrackQuality_) {
    const auto& glbQualTag = iConfig.getParameter<edm::InputTag>("globalTrackQualityInputTag");
    glbQualToken_ = consumes<edm::ValueMap<reco::MuonQuality> >(glbQualTag);
  }

  if (fillTrackerKink_) {
    trackerKinkFinder_ =
        std::make_unique<MuonKinkFinder>(iConfig.getParameter<edm::ParameterSet>("TrackerKinkFinderParameters"), iC);
  }

  if (selectHighPurity_) {
    const auto& pvTag = iConfig.getParameter<edm::InputTag>("pvInputTag");
    pvToken_ = mayConsume<reco::VertexCollection>(pvTag);
  }

  //create mesh holder
  meshAlgo_ = std::make_unique<MuonMesh>(iConfig.getParameter<edm::ParameterSet>("arbitrationCleanerOptions"));

  edm::InputTag rpcHitTag("rpcRecHits");
  rpcHitToken_ = consumes<RPCRecHitCollection>(rpcHitTag);

  edm::InputTag gemHitTag("gemRecHits");
  gemHitToken_ = consumes<GEMRecHitCollection>(gemHitTag);

  //Consumes... UGH
  inputCollectionTypes_.resize(inputCollectionLabels_.size());
  for (unsigned int i = 0; i < inputCollectionLabels_.size(); ++i) {
    const auto inputLabel = inputCollectionLabels_[i];
    const auto inputType = ICTypes::toKey(inputCollectionTypes[i]);  // Note: thorws exception if type is undefined.

    if (inputType == ICTypes::INNER_TRACKS) {
      innerTrackCollectionToken_ = consumes<reco::TrackCollection>(inputLabel);
    } else if (inputType == ICTypes::OUTER_TRACKS) {
      outerTrackCollectionToken_ = consumes<reco::TrackCollection>(inputLabel);
    } else if (inputType == ICTypes::LINKS) {
      linkCollectionToken_ = consumes<reco::MuonTrackLinksCollection>(inputLabel);
    } else if (inputType == ICTypes::MUONS) {
      muonCollectionToken_ = consumes<reco::MuonCollection>(inputLabel);
    } else if (inputType == ICTypes::TEV_FIRSTHIT) {
      tpfmsCollectionToken_ = consumes<reco::TrackToTrackMap>(inputLabel);
    } else if (fillGlobalTrackRefits_ && inputType == ICTypes::TEV_PICKY) {
      pickyCollectionToken_ = consumes<reco::TrackToTrackMap>(inputLabel);
    } else if (fillGlobalTrackRefits_ && inputType == ICTypes::TEV_DYT) {
      dytCollectionToken_ = consumes<reco::TrackToTrackMap>(inputCollectionLabels_.at(i));
    }

    inputCollectionTypes_[i] = inputType;
  }
}

MuonIdProducer::~MuonIdProducer() {
  // TimingReport::current()->dump(std::cout);
}

void MuonIdProducer::init(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  innerTrackCollectionHandle_.clear();
  outerTrackCollectionHandle_.clear();
  linkCollectionHandle_.clear();
  muonCollectionHandle_.clear();

  tpfmsCollectionHandle_.clear();
  pickyCollectionHandle_.clear();
  dytCollectionHandle_.clear();

  trackAssociator_.setPropagator(&iSetup.getData(propagatorToken_));

  if (fillTrackerKink_)
    trackerKinkFinder_->init(iSetup);

  for (unsigned int i = 0; i < inputCollectionLabels_.size(); ++i) {
    const auto& inputLabel = inputCollectionLabels_[i];
    const auto inputType = inputCollectionTypes_[i];
    if (inputType == ICTypes::INNER_TRACKS) {
      iEvent.getByToken(innerTrackCollectionToken_, innerTrackCollectionHandle_);
      if (!innerTrackCollectionHandle_.isValid())
        throw cms::Exception("FatalError") << "Failed to get input track collection with label: " << inputLabel;
      LogTrace("MuonIdentification") << "Number of input inner tracks: " << innerTrackCollectionHandle_->size();
    } else if (inputType == ICTypes::OUTER_TRACKS) {
      iEvent.getByToken(outerTrackCollectionToken_, outerTrackCollectionHandle_);
      if (!outerTrackCollectionHandle_.isValid())
        throw cms::Exception("FatalError") << "Failed to get input track collection with label: " << inputLabel;
      LogTrace("MuonIdentification") << "Number of input outer tracks: " << outerTrackCollectionHandle_->size();
    } else if (inputType == ICTypes::LINKS) {
      iEvent.getByToken(linkCollectionToken_, linkCollectionHandle_);
      if (!linkCollectionHandle_.isValid())
        throw cms::Exception("FatalError") << "Failed to get input link collection with label: " << inputLabel;
      LogTrace("MuonIdentification") << "Number of input links: " << linkCollectionHandle_->size();
    } else if (inputType == ICTypes::MUONS) {
      iEvent.getByToken(muonCollectionToken_, muonCollectionHandle_);
      if (!muonCollectionHandle_.isValid())
        throw cms::Exception("FatalError") << "Failed to get input muon collection with label: " << inputLabel;
      LogTrace("MuonIdentification") << "Number of input muons: " << muonCollectionHandle_->size();
    } else if (fillGlobalTrackRefits_ && inputType == ICTypes::TEV_FIRSTHIT) {
      iEvent.getByToken(tpfmsCollectionToken_, tpfmsCollectionHandle_);
      if (!tpfmsCollectionHandle_.isValid())
        throw cms::Exception("FatalError") << "Failed to get input muon collection with label: " << inputLabel;
      LogTrace("MuonIdentification") << "Number of input muons: " << tpfmsCollectionHandle_->size();
    } else if (fillGlobalTrackRefits_ && inputType == ICTypes::TEV_PICKY) {
      iEvent.getByToken(pickyCollectionToken_, pickyCollectionHandle_);
      if (!pickyCollectionHandle_.isValid())
        throw cms::Exception("FatalError") << "Failed to get input muon collection with label: " << inputLabel;
      LogTrace("MuonIdentification") << "Number of input muons: " << pickyCollectionHandle_->size();
    } else if (fillGlobalTrackRefits_ && inputType == ICTypes::TEV_DYT) {
      iEvent.getByToken(dytCollectionToken_, dytCollectionHandle_);
      if (!dytCollectionHandle_.isValid())
        throw cms::Exception("FatalError") << "Failed to get input muon collection with label: " << inputLabel;
      LogTrace("MuonIdentification") << "Number of input muons: " << dytCollectionHandle_->size();
    } else
      throw cms::Exception("FatalError") << "Unknown input collection type: #" << ICTypes::toStr(inputType);
  }

  iEvent.getByToken(rpcHitToken_, rpcHitHandle_);
  iEvent.getByToken(gemHitToken_, gemHitHandle_);
  if (fillGlobalTrackQuality_)
    iEvent.getByToken(glbQualToken_, glbQualHandle_);
  if (selectHighPurity_)
    iEvent.getByToken(pvToken_, pvHandle_);
}

reco::Muon MuonIdProducer::makeMuon(edm::Event& iEvent,
                                    const edm::EventSetup& iSetup,
                                    const reco::TrackRef& track,
                                    MuonIdProducer::TrackType type) {
  LogTrace("MuonIdentification") << "Creating a muon from a track " << track.get()->pt()
                                 << " Pt (GeV), eta: " << track.get()->eta();
  reco::Muon aMuon(makeMuon(*(track.get())));

  LogTrace("MuonIdentification") << "Muon created from a track ";

  aMuon.setMuonTrack(type, track);
  aMuon.setBestTrack(type);
  aMuon.setTunePBestTrack(type);

  LogTrace("MuonIdentification")
      << "Muon created from a track and setMuonBestTrack, setBestTrack and setTunePBestTrack called";

  return aMuon;
}

reco::CaloMuon MuonIdProducer::makeCaloMuon(const reco::Muon& muon) {
  LogTrace("MuonIdentification") << "Creating a CaloMuon from a Muon";

  reco::CaloMuon aMuon;
  aMuon.setInnerTrack(muon.innerTrack());

  if (muon.isEnergyValid())
    aMuon.setCalEnergy(muon.calEnergy());
  // get calo compatibility
  if (fillCaloCompatibility_)
    aMuon.setCaloCompatibility(muonCaloCompatibility_.evaluate(muon));
  return aMuon;
}

reco::Muon MuonIdProducer::makeMuon(const reco::MuonTrackLinks& links) {
  LogTrace("MuonIdentification") << "Creating a muon from a link to tracks object";

  reco::Muon aMuon;
  reco::Muon::MuonTrackTypePair chosenTrack;
  reco::TrackRef tpfmsRef;
  reco::TrackRef pickyRef;
  reco::TrackRef dytRef;
  bool useSigmaSwitch = false;

  if (tpfmsCollectionHandle_.isValid() && !tpfmsCollectionHandle_.failedToGet() && pickyCollectionHandle_.isValid() &&
      !pickyCollectionHandle_.failedToGet()) {
    tpfmsRef = muon::getTevRefitTrack(links.globalTrack(), *tpfmsCollectionHandle_);
    pickyRef = muon::getTevRefitTrack(links.globalTrack(), *pickyCollectionHandle_);
    dytRef = muon::getTevRefitTrack(links.globalTrack(), *dytCollectionHandle_);

    if (tpfmsRef.isNull() && pickyRef.isNull() && dytRef.isNull()) {
      edm::LogWarning("MakeMuonWithTEV") << "Failed to get  TEV refits, fall back to sigma switch.";
      useSigmaSwitch = true;
    }
  } else {
    useSigmaSwitch = true;
  }

  if (useSigmaSwitch) {
    chosenTrack = muon::sigmaSwitch(links.globalTrack(),
                                    links.trackerTrack(),
                                    sigmaThresholdToFillCandidateP4WithGlobalFit_,
                                    ptThresholdToFillCandidateP4WithGlobalFit_);
  } else {
    chosenTrack = muon::tevOptimized(links.globalTrack(),
                                     links.trackerTrack(),
                                     tpfmsRef,
                                     pickyRef,
                                     dytRef,
                                     ptThresholdToFillCandidateP4WithGlobalFit_);
  }
  aMuon = makeMuon(*chosenTrack.first);
  aMuon.setInnerTrack(links.trackerTrack());
  aMuon.setOuterTrack(links.standAloneTrack());
  aMuon.setGlobalTrack(links.globalTrack());
  aMuon.setBestTrack(chosenTrack.second);
  aMuon.setTunePBestTrack(chosenTrack.second);

  if (fillGlobalTrackRefits_) {
    if (tpfmsCollectionHandle_.isValid() && !tpfmsCollectionHandle_.failedToGet()) {
      reco::TrackToTrackMap::const_iterator it = tpfmsCollectionHandle_->find(links.globalTrack());
      if (it != tpfmsCollectionHandle_->end())
        aMuon.setMuonTrack(reco::Muon::TPFMS, (it->val));
    }
    if (pickyCollectionHandle_.isValid() && !pickyCollectionHandle_.failedToGet()) {
      reco::TrackToTrackMap::const_iterator it = pickyCollectionHandle_->find(links.globalTrack());
      if (it != pickyCollectionHandle_->end())
        aMuon.setMuonTrack(reco::Muon::Picky, (it->val));
    }
    if (dytCollectionHandle_.isValid() && !dytCollectionHandle_.failedToGet()) {
      reco::TrackToTrackMap::const_iterator it = dytCollectionHandle_->find(links.globalTrack());
      if (it != dytCollectionHandle_->end())
        aMuon.setMuonTrack(reco::Muon::DYT, (it->val));
    }
  }
  return aMuon;
}

bool MuonIdProducer::isGoodTrack(const reco::Track& track) {
  // Pt and absolute momentum requirement
  const double p = track.p();
  const double pt = track.pt();
  if (pt < minPt_ || (p < minP_ && p < minPCaloMuon_)) {
    LogTrace("MuonIdentification") << "Skipped low momentum track (Pt,P): " << pt << ", " << track.p() << " GeV";
    return false;
  }

  // Eta requirement
  const double eta = track.eta();
  const double absEta = std::abs(eta);
  if (absEta > maxAbsEta_) {
    LogTrace("MuonIdentification") << "Skipped track with large pseudo rapidity (Eta: " << track.eta() << " )";
    return false;
  }

  return true;
}

unsigned int MuonIdProducer::chamberId(const DetId& id) {
  if (id.det() != DetId::Muon)
    return 0;

  const auto subdetId = id.subdetId();
  if (subdetId == MuonSubdetId::DT) {
    return DTChamberId(id.rawId()).rawId();
  } else if (subdetId == MuonSubdetId::CSC) {
    return CSCDetId(id.rawId()).chamberId().rawId();
  }

  return 0;
}

int MuonIdProducer::overlap(const reco::Muon& muon, const reco::Track& track) {
  if (!muon.isMatchesValid() || track.extra().isNull() || track.extra()->recHitsSize() == 0)
    return 0;

  int numberOfCommonDetIds = 0;
  const std::vector<reco::MuonChamberMatch>& matches(muon.matches());
  for (const auto& match : matches) {
    if (match.segmentMatches.empty())
      continue;

    bool foundCommonDetId = false;
    for (auto hit = track.extra()->recHitsBegin(); hit != track.extra()->recHitsEnd(); ++hit) {
      // LogTrace("MuonIdentification") << "hit DetId: " << std::hex << hit->get()->geographicalId().rawId() <<
      //  "\t hit chamber DetId: " << getChamberId(hit->get()->geographicalId()) <<
      //  "\t segment DetId: " << match->id.rawId() << std::dec;

      if (chamberId((*hit)->geographicalId()) == match.id.rawId()) {
        foundCommonDetId = true;
        break;
      }
    }
    if (foundCommonDetId) {
      ++numberOfCommonDetIds;
      break;
    }
  }
  return numberOfCommonDetIds;
}

void MuonIdProducer::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  meshAlgo_->setCSCGeometry(&iSetup.getData(geomTokenRun_));

  if (fillShowerDigis_ && fillMatching_)
    theShowerDigiFiller_->getES(iSetup);
}

bool validateGlobalMuonPair(const reco::MuonTrackLinks& goodMuon, const reco::MuonTrackLinks& badMuon) {
  const int nHitsGood = goodMuon.globalTrack()->hitPattern().numberOfValidMuonHits();
  const int nHitsBad = badMuon.globalTrack()->hitPattern().numberOfValidMuonHits();
  if (std::min(nHitsGood, nHitsBad) > 10) {
    const double chi2Good = goodMuon.globalTrack()->normalizedChi2();
    const double chi2Bad = badMuon.globalTrack()->normalizedChi2();
    return (chi2Good <= chi2Bad);
  }

  return (nHitsGood >= nHitsBad);
}

void MuonIdProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto outputMuons = std::make_unique<reco::MuonCollection>();
  auto caloMuons = std::make_unique<reco::CaloMuonCollection>();

  init(iEvent, iSetup);

  if (fillShowerDigis_ && fillMatching_)
    theShowerDigiFiller_->getDigis(iEvent);

  // loop over input collections

  // muons first - no cleaning, take as is.
  if (muonCollectionHandle_.isValid()) {
    for (const auto& muon : *muonCollectionHandle_) {
      outputMuons->push_back(muon);
    }
  }

  // links second ( assume global muon type )
  if (linkCollectionHandle_.isValid()) {
    const auto nLink = linkCollectionHandle_->size();
    std::vector<bool> goodmuons(nLink, true);
    if (nLink > 1) {
      // check for shared tracker tracks
      for (unsigned int i = 0; i < nLink - 1; ++i) {
        const auto& iLink = linkCollectionHandle_->at(i);
        if (iLink.trackerTrack().isNull() || !checkLinks(&iLink))
          continue;
        for (unsigned int j = i + 1; j < nLink; ++j) {
          const auto& jLink = linkCollectionHandle_->at(j);
          if (!checkLinks(&jLink))
            continue;
          if (iLink.trackerTrack() == jLink.trackerTrack()) {
            // Tracker track is the essential part that dominates muon resolution
            // so taking either muon is fine. All that is important is to preserve
            // the muon identification information. If number of hits is small,
            // keep the one with large number of hits, otherwise take the smalest chi2/ndof
            if (validateGlobalMuonPair(iLink, jLink))
              goodmuons[j] = false;
            else
              goodmuons[i] = false;
          }
        }
      }
      // check for shared stand-alone muons.
      for (unsigned int i = 0; i < nLink - 1; ++i) {
        if (!goodmuons[i])
          continue;
        const auto& iLink = linkCollectionHandle_->at(i);
        if (iLink.standAloneTrack().isNull() || !checkLinks(&iLink))
          continue;
        for (unsigned int j = i + 1; j < nLink; ++j) {
          if (!goodmuons[j])
            continue;
          const auto& jLink = linkCollectionHandle_->at(j);
          if (!checkLinks(&jLink))
            continue;
          if (iLink.standAloneTrack() == jLink.standAloneTrack()) {
            if (validateGlobalMuonPair(iLink, jLink))
              goodmuons[j] = false;
            else
              goodmuons[i] = false;
          }
        }
      }
    }
    for (unsigned int i = 0; i < nLink; ++i) {
      if (!goodmuons[i])
        continue;
      const auto& iLink = linkCollectionHandle_->at(i);
      if (!checkLinks(&iLink))
        continue;
      // check if this muon is already in the list
      bool newMuon = true;
      for (const auto& muon : *outputMuons) {
        if (muon.track() == iLink.trackerTrack() && muon.standAloneMuon() == iLink.standAloneTrack() &&
            muon.combinedMuon() == iLink.globalTrack()) {
          newMuon = false;
          break;
        }
      }
      if (newMuon) {
        outputMuons->push_back(makeMuon(iLink));
        outputMuons->back().setType(reco::Muon::GlobalMuon | reco::Muon::StandAloneMuon);
      }
    }
  }

  // tracker and calo muons are next
  if (innerTrackCollectionHandle_.isValid()) {
    LogTrace("MuonIdentification") << "Creating tracker muons";
    std::vector<TrackDetectorAssociator::Direction> directions1, directions2;
    directions1.push_back(TrackDetectorAssociator::InsideOut);
    directions1.push_back(TrackDetectorAssociator::OutsideIn);
    directions2.push_back(TrackDetectorAssociator::Any);

    const GlobalTrackingGeometry* geometry = nullptr;
    if (debugWithTruthMatching_) {
      geometry = &iSetup.getData(globalGeomToken_);
    }

    for (unsigned int i = 0; i < innerTrackCollectionHandle_->size(); ++i) {
      const reco::Track& track = innerTrackCollectionHandle_->at(i);
      if (!isGoodTrack(track))
        continue;
      if (selectHighPurity_ && !track.quality(reco::TrackBase::highPurity)) {
        const reco::VertexCollection* recoVertices = pvHandle_.product();
        if (!(*recoVertices)[0].isFake())
          continue;
      }
      const auto& trackRef = reco::TrackRef(innerTrackCollectionHandle_, i);
      bool splitTrack = false;
      if (track.extra().isAvailable() && TrackDetectorAssociator::crossedIP(track))
        splitTrack = true;
      const auto& directions = splitTrack ? directions1 : directions2;
      for (const auto direction : directions) {
        // make muon
        reco::Muon trackerMuon(makeMuon(iEvent, iSetup, trackRef, reco::Muon::InnerTrack));
        fillMuonId(iEvent, iSetup, trackerMuon, direction);

        if (debugWithTruthMatching_) {
          // add MC hits to a list of matched segments.
          // Since it's debugging mode - code is slow
          MuonIdTruthInfo::truthMatchMuon(iEvent, *geometry, trackerMuon);
        }

        // check if this muon is already in the list
        // have to check where muon hits are really located
        // to match properly
        bool newMuon = true;
        const bool goodTrackerMuon = isGoodTrackerMuon(trackerMuon);
        const bool goodRPCMuon = isGoodRPCMuon(trackerMuon);
        const bool goodGEMMuon = isGoodGEMMuon(trackerMuon);
        const bool goodME0Muon = isGoodME0Muon(trackerMuon);
        if (goodTrackerMuon)
          trackerMuon.setType(trackerMuon.type() | reco::Muon::TrackerMuon);
        if (goodRPCMuon)
          trackerMuon.setType(trackerMuon.type() | reco::Muon::RPCMuon);
        if (goodGEMMuon)
          trackerMuon.setType(trackerMuon.type() | reco::Muon::GEMMuon);
        if (goodME0Muon)
          trackerMuon.setType(trackerMuon.type() | reco::Muon::ME0Muon);

        for (auto& muon : *outputMuons) {
          if (muon.innerTrack().get() == trackerMuon.innerTrack().get() &&
              std::abs(reco::deltaPhi(phiOfMuonInteractionRegion(muon), phiOfMuonInteractionRegion(trackerMuon))) <
                  M_PI_2) {
            newMuon = false;
            muon.setMatches(trackerMuon.matches());
            if (trackerMuon.isTimeValid())
              muon.setTime(trackerMuon.time());
            if (trackerMuon.isEnergyValid())
              muon.setCalEnergy(trackerMuon.calEnergy());
            if (goodTrackerMuon)
              muon.setType(muon.type() | reco::Muon::TrackerMuon);
            if (goodRPCMuon)
              muon.setType(muon.type() | reco::Muon::RPCMuon);
            if (goodGEMMuon)
              muon.setType(muon.type() | reco::Muon::GEMMuon);
            if (goodME0Muon)
              muon.setType(muon.type() | reco::Muon::ME0Muon);
            LogTrace("MuonIdentification") << "Found a corresponding global muon. Set energy, matches and move on";
            break;
          }
        }
        if (newMuon) {
          if (goodTrackerMuon || goodRPCMuon || goodGEMMuon || goodME0Muon) {
            outputMuons->push_back(trackerMuon);
          } else {
            LogTrace("MuonIdentification") << "track failed minimal number of muon matches requirement";
            const reco::CaloMuon& caloMuon = makeCaloMuon(trackerMuon);
            if (isGoodCaloMuon(caloMuon))
              caloMuons->push_back(caloMuon);
          }
        }
      }
    }
  }

  // and at last the stand alone muons
  if (outerTrackCollectionHandle_.isValid()) {
    LogTrace("MuonIdentification") << "Looking for new muons among stand alone muon tracks";
    for (unsigned int i = 0; i < outerTrackCollectionHandle_->size(); ++i) {
      const auto& outerTrack = outerTrackCollectionHandle_->at(i);

      // check if this muon is already in the list of global muons
      bool newMuon = true;
      for (auto& muon : *outputMuons) {
        if (!muon.standAloneMuon().isNull()) {
          // global muon
          if (muon.standAloneMuon().get() == &outerTrack ||
              (muon.standAloneMuon()->extra().isNonnull() &&
               muon.standAloneMuon()->extra().get() == outerTrack.extra().get())) {
            newMuon = false;
            break;
          }
        } else {
          // tracker muon - no direct links to the standalone muon
          // since we have only a few real muons in an event, matching
          // the stand alone muon to the tracker muon by DetIds should
          // be good enough for association. At the end it's up to a
          // user to redefine the association and what it means. Here
          // we would like to avoid obvious double counting and we
          // tolerate a potential miss association
          if (overlap(muon, outerTrack) > 0) {
            LogTrace("MuonIdentification") << "Found associated tracker muon. Set a reference and move on";
            newMuon = false;
            muon.setOuterTrack(reco::TrackRef(outerTrackCollectionHandle_, i));
            muon.setType(muon.type() | reco::Muon::StandAloneMuon);
            break;
          }
        }
      }
      if (newMuon) {
        LogTrace("MuonIdentification") << "No associated stand alone track is found. Making a muon";
        outputMuons->push_back(
            makeMuon(iEvent, iSetup, reco::TrackRef(outerTrackCollectionHandle_, i), reco::Muon::OuterTrack));
        outputMuons->back().setType(reco::Muon::StandAloneMuon);
      }
    }
  }

  if (arbitrateTrackerMuons_) {
    fillArbitrationInfo(outputMuons.get());
    arbitrateMuons(outputMuons.get(), caloMuons.get());
  }

  LogTrace("MuonIdentification") << "Dress up muons if it's necessary";

  const int nMuons = outputMuons->size();

  std::vector<reco::MuonTimeExtra> dtTimeColl(nMuons);
  std::vector<reco::MuonTimeExtra> cscTimeColl(nMuons);
  std::vector<reco::MuonTimeExtra> combinedTimeColl(nMuons);
  std::vector<reco::IsoDeposit> trackDepColl(nMuons);
  std::vector<reco::IsoDeposit> ecalDepColl(nMuons);
  std::vector<reco::IsoDeposit> hcalDepColl(nMuons);
  std::vector<reco::IsoDeposit> hoDepColl(nMuons);
  std::vector<reco::IsoDeposit> jetDepColl(nMuons);

  // Fill various information
  for (unsigned int i = 0; i < outputMuons->size(); ++i) {
    auto& muon = outputMuons->at(i);

    // Fill muonID
    if ((fillMatching_ && !muon.isMatchesValid()) || (fillEnergy_ && !muon.isEnergyValid())) {
      // predict direction based on the muon interaction region location
      // if it's available
      if (muon.isStandAloneMuon()) {
        if (std::abs(reco::deltaPhi(phiOfMuonInteractionRegion(muon), muon.phi())) < M_PI_2) {
          fillMuonId(iEvent, iSetup, muon, TrackDetectorAssociator::InsideOut);
        } else {
          fillMuonId(iEvent, iSetup, muon, TrackDetectorAssociator::OutsideIn);
        }
      } else {
        LogTrace("MuonIdentification") << "THIS SHOULD NEVER HAPPEN";
        fillMuonId(iEvent, iSetup, muon);
      }
    }

    if (fillGlobalTrackQuality_) {
      // Fill global quality information
      fillGlbQuality(iEvent, iSetup, muon);
    }
    LogDebug("MuonIdentification") << "";

    if (fillTrackerKink_) {
      fillTrackerKink(muon);
    }

    if (fillCaloCompatibility_)
      muon.setCaloCompatibility(muonCaloCompatibility_.evaluate(muon));

    if (fillIsolation_) {
      fillMuonIsolation(
          iEvent, iSetup, muon, trackDepColl[i], ecalDepColl[i], hcalDepColl[i], hoDepColl[i], jetDepColl[i]);
    }

    // fill timing information
    reco::MuonTime muonTime;
    reco::MuonTimeExtra dtTime;
    reco::MuonTimeExtra cscTime;
    reco::MuonTime rpcTime;
    reco::MuonTimeExtra combinedTime;

    theTimingFiller_->fillTiming(muon, dtTime, cscTime, rpcTime, combinedTime, iEvent, iSetup);

    muonTime.nDof = combinedTime.nDof();
    muonTime.timeAtIpInOut = combinedTime.timeAtIpInOut();
    muonTime.timeAtIpInOutErr = combinedTime.timeAtIpInOutErr();
    muonTime.timeAtIpOutIn = combinedTime.timeAtIpOutIn();
    muonTime.timeAtIpOutInErr = combinedTime.timeAtIpOutInErr();

    muon.setTime(muonTime);
    muon.setRPCTime(rpcTime);
    dtTimeColl[i] = dtTime;
    cscTimeColl[i] = cscTime;
    combinedTimeColl[i] = combinedTime;
  }

  LogTrace("MuonIdentification") << "number of muons produced: " << outputMuons->size();
  if (fillMatching_) {
    fillArbitrationInfo(outputMuons.get(), reco::Muon::TrackerMuon);
    fillArbitrationInfo(outputMuons.get(), reco::Muon::ME0Muon);
    fillArbitrationInfo(outputMuons.get(), reco::Muon::GEMMuon);
  }
  edm::OrphanHandle<reco::MuonCollection> muonHandle = iEvent.put(std::move(outputMuons));

  auto fillMap = [](auto refH, auto& vec, edm::Event& ev, const std::string& cAl = "") {
    typedef edm::ValueMap<typename std::decay<decltype(vec)>::type::value_type> MapType;
    auto oMap = std::make_unique<MapType>();
    {
      typename MapType::Filler filler(*oMap);
      filler.insert(refH, vec.begin(), vec.end());
      vec.clear();
      filler.fill();
    }
    ev.put(std::move(oMap), cAl);
  };
  fillMap(muonHandle, combinedTimeColl, iEvent, "combined");
  fillMap(muonHandle, dtTimeColl, iEvent, "dt");
  fillMap(muonHandle, cscTimeColl, iEvent, "csc");

  if (writeIsoDeposits_ && fillIsolation_) {
    fillMap(muonHandle, trackDepColl, iEvent, trackDepositName_);
    fillMap(muonHandle, ecalDepColl, iEvent, ecalDepositName_);
    fillMap(muonHandle, hcalDepColl, iEvent, hcalDepositName_);
    fillMap(muonHandle, hoDepColl, iEvent, hoDepositName_);
    fillMap(muonHandle, jetDepColl, iEvent, jetDepositName_);
  }

  iEvent.put(std::move(caloMuons));
}

bool MuonIdProducer::isGoodTrackerMuon(const reco::Muon& muon) {
  if (muon.track()->pt() < minPt_ || muon.track()->p() < minP_)
    return false;
  if (addExtraSoftMuons_ && muon.pt() < 5 && std::abs(muon.eta()) < 1.5 &&
      muon.numberOfMatches(reco::Muon::NoArbitration) >= 1)
    return true;
  return (muon.numberOfMatches(reco::Muon::NoArbitration) >= minNumberOfMatches_);
}

bool MuonIdProducer::isGoodCaloMuon(const reco::CaloMuon& caloMuon) {
  if (!caloMuon.isCaloCompatibilityValid() || caloMuon.caloCompatibility() < caloCut_ || caloMuon.p() < minPCaloMuon_)
    return false;
  return true;
}

bool MuonIdProducer::isGoodRPCMuon(const reco::Muon& muon) {
  if (muon.track()->pt() < minPt_ || muon.track()->p() < minP_)
    return false;
  if (addExtraSoftMuons_ && muon.pt() < 5 && std::abs(muon.eta()) < 1.5 &&
      muon.numberOfMatchedRPCLayers(reco::Muon::RPCHitAndTrackArbitration) > 1)
    return true;
  return (muon.numberOfMatchedRPCLayers(reco::Muon::RPCHitAndTrackArbitration) > minNumberOfMatches_);
}

bool MuonIdProducer::isGoodGEMMuon(const reco::Muon& muon) {
  // require GEMMuon to be a TrackerMuon
  if (!isGoodTrackerMuon(muon))
    return false;
  if (muon.track()->pt() < minPt_ || muon.track()->p() < minP_)
    return false;
  return (muon.numberOfMatches(reco::Muon::GEMSegmentAndTrackArbitration) +
          muon.numberOfMatches(reco::Muon::GEMHitAndTrackArbitration)) >= 1;
}

bool MuonIdProducer::isGoodME0Muon(const reco::Muon& muon) {
  // need to update min cuts on pt
  if (muon.track()->p() < minP_)
    return false;
  return (muon.numberOfMatches(reco::Muon::ME0SegmentAndTrackArbitration) >= 1);
}

void MuonIdProducer::fillMuonId(edm::Event& iEvent,
                                const edm::EventSetup& iSetup,
                                reco::Muon& aMuon,
                                TrackDetectorAssociator::Direction direction) {
  LogTrace("MuonIdentification") << "RecoMuon/MuonIdProducer :: fillMuonId";

  // perform track - detector association
  const reco::Track* track = nullptr;
  if (aMuon.track().isNonnull())
    track = aMuon.track().get();
  else if (aMuon.standAloneMuon().isNonnull())
    track = aMuon.standAloneMuon().get();
  else
    throw cms::Exception("FatalError")
        << "Failed to fill muon id information for a muon with undefined references to tracks";

  TrackDetMatchInfo info = trackAssociator_.associate(iEvent, iSetup, *track, parameters_, direction);

  LogTrace("MuonIdentification") << "RecoMuon/MuonIdProducer :: fillMuonId :: fillEnergy = " << fillEnergy_;

  if (fillEnergy_) {
    reco::MuonEnergy muonEnergy;
    muonEnergy.em = info.crossedEnergy(TrackDetMatchInfo::EcalRecHits);
    muonEnergy.had = info.crossedEnergy(TrackDetMatchInfo::HcalRecHits);
    muonEnergy.ho = info.crossedEnergy(TrackDetMatchInfo::HORecHits);
    muonEnergy.tower = info.crossedEnergy(TrackDetMatchInfo::TowerTotal);
    muonEnergy.emS9 = info.nXnEnergy(TrackDetMatchInfo::EcalRecHits, 1);    // 3x3 energy
    muonEnergy.emS25 = info.nXnEnergy(TrackDetMatchInfo::EcalRecHits, 2);   // 5x5 energy
    muonEnergy.hadS9 = info.nXnEnergy(TrackDetMatchInfo::HcalRecHits, 1);   // 3x3 energy
    muonEnergy.hoS9 = info.nXnEnergy(TrackDetMatchInfo::HORecHits, 1);      // 3x3 energy
    muonEnergy.towerS9 = info.nXnEnergy(TrackDetMatchInfo::TowerTotal, 1);  // 3x3 energy
    if (storeCrossedHcalRecHits_) {
      muonEnergy.crossedHadRecHits.clear();
      for (auto hit : info.crossedHcalRecHits) {
        reco::HcalMuonRecHit mhit;
        mhit.energy = hit->energy();
        mhit.chi2 = hit->chi2();
        mhit.time = hit->time();
        mhit.detId = hit->id();
        muonEnergy.crossedHadRecHits.push_back(mhit);
      }
    }
    muonEnergy.ecal_position = info.trkGlobPosAtEcal;
    muonEnergy.hcal_position = info.trkGlobPosAtHcal;
    if (!info.crossedEcalIds.empty())
      muonEnergy.ecal_id = info.crossedEcalIds.front();
    if (!info.crossedHcalIds.empty())
      muonEnergy.hcal_id = info.crossedHcalIds.front();
    // find maximal energy depositions and their time
    DetId emMaxId = info.findMaxDeposition(TrackDetMatchInfo::EcalRecHits, 2);  // max energy deposit in 5x5 shape
    for (const auto& hit : info.ecalRecHits) {
      if (hit->id() != emMaxId)
        continue;
      muonEnergy.emMax = hit->energy();
      muonEnergy.ecal_time = hit->time();
    }
    DetId hadMaxId = info.findMaxDeposition(TrackDetMatchInfo::HcalRecHits, 1);  // max energy deposit in 3x3 shape
    for (const auto& hit : info.hcalRecHits) {
      if (hit->id() != hadMaxId)
        continue;
      muonEnergy.hadMax = hit->energy();
      muonEnergy.hcal_time = hit->time();
    }
    aMuon.setCalEnergy(muonEnergy);
  }
  if (!fillMatching_ && !aMuon.isTrackerMuon() && !aMuon.isRPCMuon() && !aMuon.isGEMMuon())
    return;

  // fill muon match info
  LogTrace("MuonIdentification") << "RecoMuon/MuonIdProducer :: fillMuonId :: fill muon match info ";
  std::vector<reco::MuonChamberMatch> muonChamberMatches;
  unsigned int nubmerOfMatchesAccordingToTrackAssociator = 0;
  for (const auto& chamber : info.chambers) {
    if (chamber.id.subdetId() == MuonSubdetId::RPC && rpcHitHandle_.isValid())
      continue;  // Skip RPC chambers, they are taken care of below)
    if (chamber.id.subdetId() == MuonSubdetId::GEM && gemHitHandle_.isValid() &&
        GEMDetId(chamber.id.rawId()).station() != 0)
      continue;  // Skip GE1/1 and 2/1 chambers, they are taken care of below)
    reco::MuonChamberMatch matchedChamber;

    const auto& lErr = chamber.tState.localError();
    const auto& lPos = chamber.tState.localPosition();
    const auto& lDir = chamber.tState.localDirection();

    const auto& localError = lErr.positionError();
    matchedChamber.x = lPos.x();
    matchedChamber.y = lPos.y();
    matchedChamber.xErr = sqrt(localError.xx());
    matchedChamber.yErr = sqrt(localError.yy());

    matchedChamber.dXdZ = lDir.z() != 0 ? lDir.x() / lDir.z() : 9999;
    matchedChamber.dYdZ = lDir.z() != 0 ? lDir.y() / lDir.z() : 9999;
    // DANGEROUS - compiler cannot guaranty parameters ordering
    AlgebraicSymMatrix55 trajectoryCovMatrix = lErr.matrix();
    matchedChamber.dXdZErr = trajectoryCovMatrix(1, 1) > 0 ? sqrt(trajectoryCovMatrix(1, 1)) : 0;
    matchedChamber.dYdZErr = trajectoryCovMatrix(2, 2) > 0 ? sqrt(trajectoryCovMatrix(2, 2)) : 0;

    matchedChamber.edgeX = chamber.localDistanceX;
    matchedChamber.edgeY = chamber.localDistanceY;

    matchedChamber.id = chamber.id;

    if (fillShowerDigis_ && fillMatching_) {
      theShowerDigiFiller_->fill(matchedChamber);
    } else {
      theShowerDigiFiller_->fillDefault(matchedChamber);
    }

    if (!chamber.segments.empty())
      ++nubmerOfMatchesAccordingToTrackAssociator;

    // fill segments
    for (const auto& segment : chamber.segments) {
      reco::MuonSegmentMatch matchedSegment;
      matchedSegment.x = segment.segmentLocalPosition.x();
      matchedSegment.y = segment.segmentLocalPosition.y();
      matchedSegment.dXdZ =
          segment.segmentLocalDirection.z() ? segment.segmentLocalDirection.x() / segment.segmentLocalDirection.z() : 0;
      matchedSegment.dYdZ =
          segment.segmentLocalDirection.z() ? segment.segmentLocalDirection.y() / segment.segmentLocalDirection.z() : 0;
      matchedSegment.xErr = segment.segmentLocalErrorXX > 0 ? sqrt(segment.segmentLocalErrorXX) : 0;
      matchedSegment.yErr = segment.segmentLocalErrorYY > 0 ? sqrt(segment.segmentLocalErrorYY) : 0;
      matchedSegment.dXdZErr = segment.segmentLocalErrorDxDz > 0 ? sqrt(segment.segmentLocalErrorDxDz) : 0;
      matchedSegment.dYdZErr = segment.segmentLocalErrorDyDz > 0 ? sqrt(segment.segmentLocalErrorDyDz) : 0;
      matchedSegment.t0 = segment.t0;
      matchedSegment.mask = 0;
      matchedSegment.dtSegmentRef = segment.dtSegmentRef;
      matchedSegment.cscSegmentRef = segment.cscSegmentRef;
      matchedSegment.gemSegmentRef = segment.gemSegmentRef;
      matchedSegment.me0SegmentRef = segment.me0SegmentRef;
      matchedSegment.hasZed_ = segment.hasZed;
      matchedSegment.hasPhi_ = segment.hasPhi;
      // test segment
      bool matchedX = false;
      bool matchedY = false;
      LogTrace("MuonIdentification") << " matching local x, segment x: " << matchedSegment.x
                                     << ", chamber x: " << matchedChamber.x << ", max: " << maxAbsDx_;
      LogTrace("MuonIdentification") << " matching local y, segment y: " << matchedSegment.y
                                     << ", chamber y: " << matchedChamber.y << ", max: " << maxAbsDy_;
      const double matchedSegChDx = std::abs(matchedSegment.x - matchedChamber.x);
      const double matchedSegChDy = std::abs(matchedSegment.y - matchedChamber.y);
      if (matchedSegment.xErr > 0 && matchedChamber.xErr > 0)
        LogTrace("MuonIdentification") << " xpull: "
                                       << matchedSegChDx / std::sqrt(std::pow(matchedSegment.xErr, 2) +
                                                                     std::pow(matchedChamber.xErr, 2));
      if (matchedSegment.yErr > 0 && matchedChamber.yErr > 0)
        LogTrace("MuonIdentification") << " ypull: "
                                       << matchedSegChDy / std::sqrt(std::pow(matchedSegment.yErr, 2) +
                                                                     std::pow(matchedChamber.yErr, 2));

      if (matchedSegChDx < maxAbsDx_)
        matchedX = true;
      else if (matchedSegment.xErr > 0 && matchedChamber.xErr > 0) {
        const double invMatchedSegChPullX2 = std::pow(matchedSegment.xErr, 2) + std::pow(matchedChamber.xErr, 2);
        if (matchedSegChDx * matchedSegChDx < maxAbsPullX2_ * invMatchedSegChPullX2)
          matchedX = true;
      }
      if (matchedSegChDy < maxAbsDy_)
        matchedY = true;
      else if (matchedSegment.yErr > 0 && matchedChamber.yErr > 0) {
        const double invMatchedSegChPullY2 = std::pow(matchedSegment.yErr, 2) + std::pow(matchedChamber.yErr, 2);
        if (matchedSegChDy * matchedSegChDy < maxAbsPullY2_ * invMatchedSegChPullY2)
          matchedY = true;
      }
      if (matchedX && matchedY) {
        if (matchedChamber.id.subdetId() == MuonSubdetId::ME0)
          matchedChamber.me0Matches.push_back(matchedSegment);
        else if (matchedChamber.id.subdetId() == MuonSubdetId::GEM)
          matchedChamber.gemMatches.push_back(matchedSegment);
        else
          matchedChamber.segmentMatches.push_back(matchedSegment);
      }
    }
    muonChamberMatches.push_back(matchedChamber);
  }

  // Fill RPC info
  LogTrace("MuonIdentification") << "RecoMuon/MuonIdProducer :: fillMuonId :: fill RPC info";
  if (rpcHitHandle_.isValid()) {
    for (const auto& chamber : info.chambers) {
      if (chamber.id.subdetId() != MuonSubdetId::RPC)
        continue;  // Consider RPC chambers only
      const auto& lErr = chamber.tState.localError();
      const auto& lPos = chamber.tState.localPosition();
      const auto& lDir = chamber.tState.localDirection();

      reco::MuonChamberMatch matchedChamber;

      LocalError localError = lErr.positionError();
      matchedChamber.x = lPos.x();
      matchedChamber.y = lPos.y();
      matchedChamber.xErr = sqrt(localError.xx());
      matchedChamber.yErr = sqrt(localError.yy());

      matchedChamber.dXdZ = lDir.z() != 0 ? lDir.x() / lDir.z() : 9999;
      matchedChamber.dYdZ = lDir.z() != 0 ? lDir.y() / lDir.z() : 9999;
      // DANGEROUS - compiler cannot guaranty parameters ordering
      AlgebraicSymMatrix55 trajectoryCovMatrix = lErr.matrix();
      matchedChamber.dXdZErr = trajectoryCovMatrix(1, 1) > 0 ? sqrt(trajectoryCovMatrix(1, 1)) : 0;
      matchedChamber.dYdZErr = trajectoryCovMatrix(2, 2) > 0 ? sqrt(trajectoryCovMatrix(2, 2)) : 0;

      matchedChamber.edgeX = chamber.localDistanceX;
      matchedChamber.edgeY = chamber.localDistanceY;

      theShowerDigiFiller_->fillDefault(matchedChamber);

      matchedChamber.id = chamber.id;

      for (const auto& rpcRecHit : *rpcHitHandle_) {
        reco::MuonRPCHitMatch rpcHitMatch;

        if (rpcRecHit.rawId() != chamber.id.rawId())
          continue;

        rpcHitMatch.x = rpcRecHit.localPosition().x();
        rpcHitMatch.mask = 0;
        rpcHitMatch.bx = rpcRecHit.BunchX();

        const double absDx = std::abs(rpcRecHit.localPosition().x() - chamber.tState.localPosition().x());
        if (absDx <= 20 or absDx * absDx <= 16 * localError.xx())
          matchedChamber.rpcMatches.push_back(rpcHitMatch);
      }

      muonChamberMatches.push_back(matchedChamber);
    }
  }

  // Fill GEM info
  LogTrace("MuonIdentification") << "RecoMuon/MuonIdProducer :: fillMuonId :: fill GEM info";
  if (gemHitHandle_.isValid()) {
    for (const auto& chamber : info.chambers) {
      // only GE1/1 and 2/1 are for rechits, reject station 0 and segments (layer==0 for GEMSegment)
      if (chamber.id.subdetId() != MuonSubdetId::GEM || GEMDetId(chamber.id.rawId()).station() == 0 ||
          GEMDetId(chamber.id.rawId()).layer() == 0)
        continue;  // Consider GEM chambers only
      const auto& lErr = chamber.tState.localError();
      const auto& lPos = chamber.tState.localPosition();
      const auto& lDir = chamber.tState.localDirection();

      reco::MuonChamberMatch matchedChamber;

      LocalError localError = lErr.positionError();
      matchedChamber.x = lPos.x();
      matchedChamber.y = lPos.y();
      matchedChamber.xErr = sqrt(localError.xx());
      matchedChamber.yErr = sqrt(localError.yy());

      matchedChamber.dXdZ = lDir.z() != 0 ? lDir.x() / lDir.z() : 9999;
      matchedChamber.dYdZ = lDir.z() != 0 ? lDir.y() / lDir.z() : 9999;
      // DANGEROUS - compiler cannot guaranty parameters ordering
      AlgebraicSymMatrix55 trajectoryCovMatrix = lErr.matrix();
      matchedChamber.dXdZErr = trajectoryCovMatrix(1, 1) > 0 ? sqrt(trajectoryCovMatrix(1, 1)) : 0;
      matchedChamber.dYdZErr = trajectoryCovMatrix(2, 2) > 0 ? sqrt(trajectoryCovMatrix(2, 2)) : 0;

      matchedChamber.edgeX = chamber.localDistanceX;
      matchedChamber.edgeY = chamber.localDistanceY;

      theShowerDigiFiller_->fillDefault(matchedChamber);

      matchedChamber.id = chamber.id;

      for (const auto& gemRecHit : *gemHitHandle_) {
        reco::MuonGEMHitMatch gemHitMatch;

        if (GEMDetId(gemRecHit.gemId().region(),
                     gemRecHit.gemId().ring(),
                     gemRecHit.gemId().station(),
                     gemRecHit.gemId().layer(),
                     gemRecHit.gemId().chamber(),
                     0)
                .rawId() != chamber.id.rawId())
          continue;

        gemHitMatch.x = gemRecHit.localPosition().x();
        gemHitMatch.mask = 0;
        gemHitMatch.bx = gemRecHit.BunchX();

        const double absDx = std::abs(gemRecHit.localPosition().x() - chamber.tState.localPosition().x());
        if (absDx <= 5 or absDx * absDx <= 16 * localError.xx())
          matchedChamber.gemHitMatches.push_back(gemHitMatch);
      }

      muonChamberMatches.push_back(matchedChamber);
    }
  }

  aMuon.setMatches(muonChamberMatches);

  LogTrace("MuonIdentification") << "number of muon chambers: " << aMuon.matches().size() << "\n"
                                 << "number of chambers with segments according to the associator requirements: "
                                 << nubmerOfMatchesAccordingToTrackAssociator;
  LogTrace("MuonIdentification") << "number of segment matches with the producer requirements: "
                                 << aMuon.numberOfMatches(reco::Muon::NoArbitration);

  // fillTime( iEvent, iSetup, aMuon );
}

void MuonIdProducer::arbitrateMuons(reco::MuonCollection* muons, reco::CaloMuonCollection* caloMuons) {
  reco::Muon::ArbitrationType arbitration = reco::Muon::SegmentAndTrackArbitration;
  // arbitrate TrackerMuons
  // if a muon was exclusively TrackerMuon check if it can be a calo muon
  for (reco::MuonCollection::iterator muon = muons->begin(); muon != muons->end();) {
    if (muon->isTrackerMuon()) {
      if (muon->numberOfMatches(arbitration) < minNumberOfMatches_) {
        // TrackerMuon failed arbitration
        // If not any other base type - erase the element
        // (PFMuon is not a base type)
        // GEMMuon should be a subset of TrackerMuon, so don't count it either
        unsigned int mask = reco::Muon::TrackerMuon | reco::Muon::PFMuon | reco::Muon::GEMMuon;
        if ((muon->type() & (~mask)) == 0) {
          const reco::CaloMuon& caloMuon = makeCaloMuon(*muon);
          if (isGoodCaloMuon(caloMuon))
            caloMuons->push_back(caloMuon);
          muon = muons->erase(muon);
          continue;
        } else {
          muon->setType(muon->type() & (~(reco::Muon::TrackerMuon | reco::Muon::GEMMuon)));
        }
      }
    }
    muon++;
  }
}

void MuonIdProducer::fillArbitrationInfo(reco::MuonCollection* pOutputMuons, unsigned int muonType) {
  //
  // apply segment flags
  //
  std::vector<std::pair<reco::MuonChamberMatch*, reco::MuonSegmentMatch*> > chamberPairs;  // for chamber segment sorting
  std::vector<std::pair<reco::MuonChamberMatch*, reco::MuonSegmentMatch*> > stationPairs;  // for station segment sorting
  std::vector<std::pair<reco::MuonChamberMatch*, reco::MuonSegmentMatch*> >
      arbitrationPairs;  // for muon segment arbitration

  // muonIndex1
  for (unsigned int muonIndex1 = 0; muonIndex1 < pOutputMuons->size(); ++muonIndex1) {
    auto& muon1 = pOutputMuons->at(muonIndex1);
    // chamberIter1
    for (auto& chamber1 : muon1.matches()) {
      // segmentIter1
      std::vector<reco::MuonSegmentMatch>* segmentMatches1 = getSegmentMatches(chamber1, muonType);

      if (segmentMatches1->empty())
        continue;
      chamberPairs.clear();

      for (auto& segment1 : *segmentMatches1) {
        chamberPairs.push_back(std::make_pair(&chamber1, &segment1));
        if (!segment1.isMask())  // has not yet been arbitrated
        {
          arbitrationPairs.clear();
          arbitrationPairs.push_back(std::make_pair(&chamber1, &segment1));

          // find identical segments with which to arbitrate
          // tracker muons only
          if (muon1.type() & muonType) {
            // muonIndex2
            for (unsigned int muonIndex2 = muonIndex1 + 1; muonIndex2 < pOutputMuons->size(); ++muonIndex2) {
              auto& muon2 = pOutputMuons->at(muonIndex2);
              // tracker muons only
              if (!(muon2.type() & muonType))
                continue;
              // chamberIter2
              for (auto& chamber2 : muon2.matches()) {
                // segmentIter2
                std::vector<reco::MuonSegmentMatch>* segmentMatches2 = getSegmentMatches(chamber2, muonType);
                for (auto& segment2 : *segmentMatches2) {
                  if (segment2.isMask())
                    continue;  // has already been arbitrated
                  if (approxEqual(segment2.x, segment1.x) && approxEqual(segment2.y, segment1.y) &&
                      approxEqual(segment2.dXdZ, segment1.dXdZ) && approxEqual(segment2.dYdZ, segment1.dYdZ) &&
                      approxEqual(segment2.xErr, segment1.xErr) && approxEqual(segment2.yErr, segment1.yErr) &&
                      approxEqual(segment2.dXdZErr, segment1.dXdZErr) &&
                      approxEqual(segment2.dYdZErr, segment1.dYdZErr)) {
                    arbitrationPairs.push_back(std::make_pair(&chamber2, &segment2));
                  }
                }  // segmentIter2
              }    // chamberIter2
            }      // muonIndex2
          }

          // arbitration segment sort
          if (arbitrationPairs.empty())
            continue;  // this should never happen
          if (arbitrationPairs.size() == 1) {
            arbitrationPairs.front().second->setMask(reco::MuonSegmentMatch::BelongsToTrackByDRSlope);
            arbitrationPairs.front().second->setMask(reco::MuonSegmentMatch::BelongsToTrackByDXSlope);
            arbitrationPairs.front().second->setMask(reco::MuonSegmentMatch::BelongsToTrackByDR);
            arbitrationPairs.front().second->setMask(reco::MuonSegmentMatch::BelongsToTrackByDX);
            arbitrationPairs.front().second->setMask(reco::MuonSegmentMatch::Arbitrated);
          } else {
            sort(arbitrationPairs.begin(),
                 arbitrationPairs.end(),
                 SortMuonSegmentMatches(reco::MuonSegmentMatch::BelongsToTrackByDRSlope));
            arbitrationPairs.front().second->setMask(reco::MuonSegmentMatch::BelongsToTrackByDRSlope);
            sort(arbitrationPairs.begin(),
                 arbitrationPairs.end(),
                 SortMuonSegmentMatches(reco::MuonSegmentMatch::BelongsToTrackByDXSlope));
            arbitrationPairs.front().second->setMask(reco::MuonSegmentMatch::BelongsToTrackByDXSlope);
            sort(arbitrationPairs.begin(),
                 arbitrationPairs.end(),
                 SortMuonSegmentMatches(reco::MuonSegmentMatch::BelongsToTrackByDR));
            arbitrationPairs.front().second->setMask(reco::MuonSegmentMatch::BelongsToTrackByDR);
            sort(arbitrationPairs.begin(),
                 arbitrationPairs.end(),
                 SortMuonSegmentMatches(reco::MuonSegmentMatch::BelongsToTrackByDX));
            arbitrationPairs.front().second->setMask(reco::MuonSegmentMatch::BelongsToTrackByDX);
            for (auto& ap : arbitrationPairs) {
              ap.second->setMask(reco::MuonSegmentMatch::Arbitrated);
            }
          }
        }

        // setup me1a cleaning for later
        if (muonType == reco::Muon::TrackerMuon && chamber1.id.subdetId() == MuonSubdetId::CSC && arbClean_ &&
            CSCDetId(chamber1.id).ring() == 4) {
          for (auto& segment2 : chamber1.segmentMatches) {
            if (segment1.cscSegmentRef.isNull() || segment2.cscSegmentRef.isNull())
              continue;
            if (meshAlgo_->isDuplicateOf(segment1.cscSegmentRef, segment2.cscSegmentRef) &&
                (segment2.mask & 0x1e0000) && (segment1.mask & 0x1e0000)) {
              segment2.setMask(reco::MuonSegmentMatch::BelongsToTrackByME1aClean);
              //if the track has lost the segment already through normal arbitration no need to do it again.
            }
          }
        }  // mark all ME1/a duplicates that this track owns

      }  // segmentIter1

      // chamber segment sort
      if (chamberPairs.empty())
        continue;  // this should never happen
      if (chamberPairs.size() == 1) {
        chamberPairs.front().second->setMask(reco::MuonSegmentMatch::BestInChamberByDRSlope);
        chamberPairs.front().second->setMask(reco::MuonSegmentMatch::BestInChamberByDXSlope);
        chamberPairs.front().second->setMask(reco::MuonSegmentMatch::BestInChamberByDR);
        chamberPairs.front().second->setMask(reco::MuonSegmentMatch::BestInChamberByDX);
      } else {
        sort(chamberPairs.begin(),
             chamberPairs.end(),
             SortMuonSegmentMatches(reco::MuonSegmentMatch::BestInChamberByDRSlope));
        chamberPairs.front().second->setMask(reco::MuonSegmentMatch::BestInChamberByDRSlope);
        sort(chamberPairs.begin(),
             chamberPairs.end(),
             SortMuonSegmentMatches(reco::MuonSegmentMatch::BestInChamberByDXSlope));
        chamberPairs.front().second->setMask(reco::MuonSegmentMatch::BestInChamberByDXSlope);
        sort(chamberPairs.begin(),
             chamberPairs.end(),
             SortMuonSegmentMatches(reco::MuonSegmentMatch::BestInChamberByDR));
        chamberPairs.front().second->setMask(reco::MuonSegmentMatch::BestInChamberByDR);
        sort(chamberPairs.begin(),
             chamberPairs.end(),
             SortMuonSegmentMatches(reco::MuonSegmentMatch::BestInChamberByDX));
        chamberPairs.front().second->setMask(reco::MuonSegmentMatch::BestInChamberByDX);
      }
    }  // chamberIter1

    // station segment sort
    for (int stationIndex = 1; stationIndex < 5; ++stationIndex) {
      for (int detectorIndex = 1; detectorIndex <= 5;
           ++detectorIndex)  // 1-5, as in DataFormats/MuonDetId/interface/MuonSubdetId.h
      {
        stationPairs.clear();

        // chamberIter
        for (auto& chamber : muon1.matches()) {
          if (!(chamber.station() == stationIndex && chamber.detector() == detectorIndex))
            continue;
          std::vector<reco::MuonSegmentMatch>* segmentMatches = getSegmentMatches(chamber, muonType);
          if (segmentMatches->empty())
            continue;

          for (auto& segment : *segmentMatches) {
            stationPairs.push_back(std::make_pair(&chamber, &segment));
          }
        }  // chamberIter

        if (stationPairs.empty())
          continue;  // this may very well happen
        if (stationPairs.size() == 1) {
          stationPairs.front().second->setMask(reco::MuonSegmentMatch::BestInStationByDRSlope);
          stationPairs.front().second->setMask(reco::MuonSegmentMatch::BestInStationByDXSlope);
          stationPairs.front().second->setMask(reco::MuonSegmentMatch::BestInStationByDR);
          stationPairs.front().second->setMask(reco::MuonSegmentMatch::BestInStationByDX);
        } else {
          sort(stationPairs.begin(),
               stationPairs.end(),
               SortMuonSegmentMatches(reco::MuonSegmentMatch::BestInStationByDRSlope));
          stationPairs.front().second->setMask(reco::MuonSegmentMatch::BestInStationByDRSlope);
          sort(stationPairs.begin(),
               stationPairs.end(),
               SortMuonSegmentMatches(reco::MuonSegmentMatch::BestInStationByDXSlope));
          stationPairs.front().second->setMask(reco::MuonSegmentMatch::BestInStationByDXSlope);
          sort(stationPairs.begin(),
               stationPairs.end(),
               SortMuonSegmentMatches(reco::MuonSegmentMatch::BestInStationByDR));
          stationPairs.front().second->setMask(reco::MuonSegmentMatch::BestInStationByDR);
          sort(stationPairs.begin(),
               stationPairs.end(),
               SortMuonSegmentMatches(reco::MuonSegmentMatch::BestInStationByDX));
          stationPairs.front().second->setMask(reco::MuonSegmentMatch::BestInStationByDX);
        }
      }
    }

  }  // muonIndex1

  if (arbClean_) {
    // clear old mesh, create and prune new mesh!
    meshAlgo_->clearMesh();
    meshAlgo_->runMesh(pOutputMuons);
  }
}

void MuonIdProducer::fillMuonIsolation(edm::Event& iEvent,
                                       const edm::EventSetup& iSetup,
                                       reco::Muon& aMuon,
                                       reco::IsoDeposit& trackDep,
                                       reco::IsoDeposit& ecalDep,
                                       reco::IsoDeposit& hcalDep,
                                       reco::IsoDeposit& hoDep,
                                       reco::IsoDeposit& jetDep) {
  const reco::Track* track = nullptr;
  if (aMuon.track().isNonnull())
    track = aMuon.track().get();
  else if (aMuon.standAloneMuon().isNonnull())
    track = aMuon.standAloneMuon().get();
  else
    throw cms::Exception("FatalError")
        << "Failed to compute muon isolation information for a muon with undefined references to tracks";

  reco::MuonIsolation isoR03, isoR05;

  // get deposits
  reco::IsoDeposit depTrk = muIsoExtractorTrack_->deposit(iEvent, iSetup, *track);
  std::vector<reco::IsoDeposit> caloDeps = muIsoExtractorCalo_->deposits(iEvent, iSetup, *track);
  reco::IsoDeposit depJet = muIsoExtractorJet_->deposit(iEvent, iSetup, *track);

  if (caloDeps.size() != 3) {
    LogTrace("MuonIdentification") << "Failed to fill vector of calorimeter isolation deposits!";
    return;
  }

  reco::IsoDeposit depEcal = caloDeps.at(0);
  reco::IsoDeposit depHcal = caloDeps.at(1);
  reco::IsoDeposit depHo = caloDeps.at(2);

  //no need to copy outside if we don't write them
  if (writeIsoDeposits_) {
    trackDep = depTrk;
    ecalDep = depEcal;
    hcalDep = depHcal;
    hoDep = depHo;
    jetDep = depJet;
  }

  isoR03.sumPt = depTrk.depositWithin(0.3);
  isoR03.emEt = depEcal.depositWithin(0.3);
  isoR03.hadEt = depHcal.depositWithin(0.3);
  isoR03.hoEt = depHo.depositWithin(0.3);
  isoR03.nTracks = depTrk.depositAndCountWithin(0.3).second;
  isoR03.nJets = depJet.depositAndCountWithin(0.3).second;
  isoR03.trackerVetoPt = depTrk.candEnergy();
  isoR03.emVetoEt = depEcal.candEnergy();
  isoR03.hadVetoEt = depHcal.candEnergy();
  isoR03.hoVetoEt = depHo.candEnergy();

  isoR05.sumPt = depTrk.depositWithin(0.5);
  isoR05.emEt = depEcal.depositWithin(0.5);
  isoR05.hadEt = depHcal.depositWithin(0.5);
  isoR05.hoEt = depHo.depositWithin(0.5);
  isoR05.nTracks = depTrk.depositAndCountWithin(0.5).second;
  isoR05.nJets = depJet.depositAndCountWithin(0.5).second;
  isoR05.trackerVetoPt = depTrk.candEnergy();
  isoR05.emVetoEt = depEcal.candEnergy();
  isoR05.hadVetoEt = depHcal.candEnergy();
  isoR05.hoVetoEt = depHo.candEnergy();

  aMuon.setIsolation(isoR03, isoR05);
}

reco::Muon MuonIdProducer::makeMuon(const reco::Track& track) {
  const double energy = std::sqrt(track.p() * track.p() + 0.105658369 * 0.105658369);
  const math::XYZTLorentzVector p4(track.px(), track.py(), track.pz(), energy);
  return reco::Muon(track.charge(), p4, track.vertex());
}

double MuonIdProducer::sectorPhi(const DetId& id) {
  double phi = 0;
  if (id.subdetId() == MuonSubdetId::DT) {  // DT
    DTChamberId muonId(id.rawId());
    if (muonId.sector() <= 12)
      phi = (muonId.sector() - 1) / 6. * M_PI;
    if (muonId.sector() == 13)
      phi = 3 / 6. * M_PI;
    if (muonId.sector() == 14)
      phi = 9 / 6. * M_PI;
  }
  if (id.subdetId() == MuonSubdetId::CSC) {  // CSC
    CSCDetId muonId(id.rawId());
    phi = M_PI / 4 + (muonId.triggerSector() - 1) / 3. * M_PI;
  }
  if (phi > M_PI)
    phi -= 2 * M_PI;
  return phi;
}

double MuonIdProducer::phiOfMuonInteractionRegion(const reco::Muon& muon) const {
  if (muon.isStandAloneMuon())
    return muon.standAloneMuon()->innerPosition().phi();
  // the rest is tracker muon only
  if (muon.matches().empty()) {
    if (muon.innerTrack().isAvailable() && muon.innerTrack()->extra().isAvailable())
      return muon.innerTrack()->outerPosition().phi();
    else
      return muon.phi();  // makes little sense, but what else can I use
  }
  return sectorPhi(muon.matches().at(0).id);
}

void MuonIdProducer::fillGlbQuality(edm::Event& iEvent, const edm::EventSetup& iSetup, reco::Muon& aMuon) {
  if (aMuon.isGlobalMuon() && glbQualHandle_.isValid() && !glbQualHandle_.failedToGet()) {
    aMuon.setCombinedQuality((*glbQualHandle_)[aMuon.combinedMuon()]);
  }

  LogDebug("MuonIdentification") << "tkChiVal " << aMuon.combinedQuality().trkRelChi2;
}

void MuonIdProducer::fillTrackerKink(reco::Muon& aMuon) {
  // skip muons with no tracks
  if (aMuon.innerTrack().isNull())
    return;
  // get quality from muon if already there, otherwise make empty one
  reco::MuonQuality quality = (aMuon.isQualityValid() ? aMuon.combinedQuality() : reco::MuonQuality());
  // fill it
  const bool filled = trackerKinkFinder_->fillTrkKink(quality, *aMuon.innerTrack());
  // if quality was there, or if we filled it, commit to the muon
  if (filled || aMuon.isQualityValid())
    aMuon.setCombinedQuality(quality);
}

bool MuonIdProducer::checkLinks(const reco::MuonTrackLinks* links) const {
  const bool trackBAD = links->trackerTrack().isNull();
  const bool staBAD = links->standAloneTrack().isNull();
  const bool glbBAD = links->globalTrack().isNull();
  if (trackBAD || staBAD || glbBAD) {
    edm::LogWarning("muonIDbadLinks") << "Global muon links to constituent tracks are invalid: trkBad " << trackBAD
                                      << " standaloneBad " << staBAD << " globalBad " << glbBAD
                                      << ". There should be no such object. Muon is skipped.";
    return false;
  }
  return true;
}

void MuonIdProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setAllowAnything();

  desc.add<bool>("arbitrateTrackerMuons", false);
  desc.add<bool>("storeCrossedHcalRecHits", false);
  desc.add<bool>("fillShowerDigis", false);
  desc.ifValue(
      edm::ParameterDescription<bool>("selectHighPurity", false, true),
      true >> (edm::ParameterDescription<edm::InputTag>("pvInputTag", edm::InputTag("offlinePrimaryVertices"), true)) or
          false >> (edm::ParameterDescription<edm::InputTag>("pvInputTag", edm::InputTag(""), true)));

  edm::ParameterSetDescription descTrkAsoPar;
  descTrkAsoPar.add<edm::InputTag>("GEMSegmentCollectionLabel", edm::InputTag("gemSegments"));
  descTrkAsoPar.add<edm::InputTag>("ME0SegmentCollectionLabel", edm::InputTag("me0Segments"));
  descTrkAsoPar.add<bool>("useGEM", false);
  descTrkAsoPar.add<bool>("useME0", false);
  descTrkAsoPar.add<bool>("preselectMuonTracks", false);
  descTrkAsoPar.add<edm::InputTag>("RPCHitCollectionLabel", edm::InputTag("rpcRecHits"));
  descTrkAsoPar.add<edm::InputTag>("GEMHitCollectionLabel", edm::InputTag("gemRecHits"));
  descTrkAsoPar.add<edm::InputTag>("ME0HitCollectionLabel", edm::InputTag("me0RecHits"));
  descTrkAsoPar.setAllowAnything();
  desc.add<edm::ParameterSetDescription>("TrackAssociatorParameters", descTrkAsoPar);

  edm::ParameterSetDescription descJet;
  descJet.setAllowAnything();
  descJet.add<edm::ParameterSetDescription>("TrackAssociatorParameters", descTrkAsoPar);
  desc.add<edm::ParameterSetDescription>("JetExtractorPSet", descJet);

  edm::ParameterSetDescription descCalo;
  descCalo.setAllowAnything();
  descCalo.add<edm::ParameterSetDescription>("TrackAssociatorParameters", descTrkAsoPar);
  desc.add<edm::ParameterSetDescription>("CaloExtractorPSet", descCalo);

  descriptions.addDefault(desc);
}

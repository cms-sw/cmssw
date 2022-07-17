#include "RecoMuon/MuonIsolationProducers/plugins/MuIsoDepositProducer.h"

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositFwd.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

#include "RecoMuon/MuonIsolation/interface/Range.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositDirection.h"

#include "PhysicsTools/IsolationAlgos/interface/IsoDepositExtractor.h"
#include "PhysicsTools/IsolationAlgos/interface/IsoDepositExtractorFactory.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <string>

using namespace edm;
using namespace std;
using namespace reco;
using namespace muonisolation;

//! constructor with config
MuIsoDepositProducer::MuIsoDepositProducer(const ParameterSet& par)
    : theDepositNames(std::vector<std::string>(1, std::string())) {
  static const std::string metname = "RecoMuon|MuonIsolationProducers|MuIsoDepositProducer";
  LogDebug(metname) << " MuIsoDepositProducer CTOR";

  edm::ParameterSet ioPSet = par.getParameter<edm::ParameterSet>("IOPSet");

  theInputType = ioPSet.getParameter<std::string>("InputType");
  theExtractForCandidate = ioPSet.getParameter<bool>("ExtractForCandidate");
  theMuonTrackRefType = ioPSet.getParameter<std::string>("MuonTrackRefType");

  bool readFromRecoTrack = theInputType == "TrackCollection";
  bool readFromRecoMuon = theInputType == "MuonCollection";
  bool readFromCandidateView = theInputType == "CandidateView";
  if (readFromRecoTrack) {
    theMuonCollectionTag = consumes<View<Track>>(ioPSet.getParameter<edm::InputTag>("inputMuonCollection"));
  } else if (readFromRecoMuon) {
    theMuonCollectionTag = consumes<View<RecoCandidate>>(ioPSet.getParameter<edm::InputTag>("inputMuonCollection"));
  } else if (readFromCandidateView) {
    theMuonCollectionTag = consumes<View<Candidate>>(ioPSet.getParameter<edm::InputTag>("inputMuonCollection"));
  } else {
    throw cms::Exception("Configuration") << "Inconsistent configuration or failure to read Candidate-muon view";
  }

  theMultipleDepositsFlag = ioPSet.getParameter<bool>("MultipleDepositsFlag");

  if (theMultipleDepositsFlag) {
    theDepositNames = par.getParameter<edm::ParameterSet>("ExtractorPSet")
                          .getParameter<std::vector<std::string>>("DepositInstanceLabels");
  }

  for (unsigned int i = 0; i < theDepositNames.size(); ++i) {
    std::string alias = par.getParameter<std::string>("@module_label");
    if (!theDepositNames[i].empty())
      alias += "_" + theDepositNames[i];
    produces<reco::IsoDepositMap>(theDepositNames[i]).setBranchAlias(alias);
  }

  edm::ParameterSet extractorPSet = par.getParameter<edm::ParameterSet>("ExtractorPSet");
  std::string extractorName = extractorPSet.getParameter<std::string>("ComponentName");
  theExtractor = IsoDepositExtractorFactory::get()->create(extractorName, extractorPSet, consumesCollector());
  LogDebug(metname) << " Load extractor..." << extractorName;
}

//! destructor
MuIsoDepositProducer::~MuIsoDepositProducer() {
  LogDebug("RecoMuon/MuIsoDepositProducer") << " MuIsoDepositProducer DTOR";
}

//! build deposits
void MuIsoDepositProducer::produce(Event& event, const EventSetup& eventSetup) {
  static const std::string metname = "RecoMuon|MuonIsolationProducers|MuIsoDepositProducer";

  LogDebug(metname) << " Muon Deposit producing..."
                    << " BEGINING OF EVENT "
                    << "================================";

  unsigned int nDeps = theMultipleDepositsFlag ? theDepositNames.size() : 1;

  // Take the muon container
  LogTrace(metname) << " Taking the muons: "
                    << theMuonCollectionTag.index();  //a more friendly print would use "inputMuonCollection"
  Handle<View<Track>> tracks;
  //! read them as RecoCandidates: need to have track() standAloneMuon() etc in the interface
  Handle<View<RecoCandidate>> muons;  //! get rid of this at some point and use the cands
  Handle<View<Candidate>> cands;

  unsigned int nMuons = 0;

  bool readFromRecoTrack = theInputType == "TrackCollection";
  bool readFromRecoMuon = theInputType == "MuonCollection";
  bool readFromCandidateView = theInputType == "CandidateView";

  if (readFromRecoMuon) {
    event.getByToken(theMuonCollectionTag, muons);
    nMuons = muons->size();
    LogDebug(metname) << "Got Muons of size " << nMuons;
  }
  if (readFromRecoTrack) {
    event.getByToken(theMuonCollectionTag, tracks);
    nMuons = tracks->size();
    LogDebug(metname) << "Got MuonTracks of size " << nMuons;
  }
  if (readFromCandidateView || theExtractForCandidate) {
    event.getByToken(theMuonCollectionTag, cands);
    unsigned int nCands = cands->size();
    if (readFromRecoMuon && theExtractForCandidate) {
      //! expect nMuons set already
      if (nMuons != nCands)
        edm::LogError(metname) << "Inconsistent configuration or failure to read Candidate-muon view";
    }
    nMuons = nCands;
    LogDebug(metname) << "Got candidate view with size " << nMuons;
  }

  static const unsigned int MAX_DEPS = 10;
  std::unique_ptr<reco::IsoDepositMap> depMaps[MAX_DEPS];

  if (nDeps > 10)
    LogError(metname) << "Unable to handle more than 10 input deposits";
  for (unsigned int i = 0; i < nDeps; ++i) {
    depMaps[i] = std::make_unique<reco::IsoDepositMap>();
  }

  //! OK, now we know how many deps for how many muons each we will create
  //! might linearize this at some point (lazy)
  //! do it in case some muons are there only
  if (nMuons > 0) {
    std::vector<std::vector<IsoDeposit>> deps2D(nDeps, std::vector<IsoDeposit>(nMuons));

    for (unsigned int i = 0; i < nMuons; ++i) {
      TrackBaseRef muRef;
      if (readFromRecoMuon) {
        if (theMuonTrackRefType == "track") {
          muRef = TrackBaseRef((*muons)[i].track());
        } else if (theMuonTrackRefType == "standAloneMuon") {
          muRef = TrackBaseRef((*muons)[i].standAloneMuon());
        } else if (theMuonTrackRefType == "combinedMuon") {
          muRef = TrackBaseRef((*muons)[i].combinedMuon());
        } else if (theMuonTrackRefType == "bestGlbTrkSta") {
          if (!(*muons)[i].combinedMuon().isNull()) {
            muRef = TrackBaseRef((*muons)[i].combinedMuon());
          } else if (!(*muons)[i].track().isNull()) {
            muRef = TrackBaseRef((*muons)[i].track());
          } else {
            muRef = TrackBaseRef((*muons)[i].standAloneMuon());
          }
        } else if (theMuonTrackRefType == "bestTrkSta") {
          if (!(*muons)[i].track().isNull()) {
            muRef = TrackBaseRef((*muons)[i].track());
          } else {
            muRef = TrackBaseRef((*muons)[i].standAloneMuon());
          }
        } else {
          edm::LogWarning(metname) << "Wrong track type is supplied: breaking";
          break;
        }
      } else if (readFromRecoTrack) {
        muRef = TrackBaseRef(tracks, i);
      }

      if (!theMultipleDepositsFlag) {
        if (readFromCandidateView || theExtractForCandidate)
          deps2D[0][i] = theExtractor->deposit(event, eventSetup, (*cands)[i]);
        else
          deps2D[0][i] = theExtractor->deposit(event, eventSetup, muRef);

      } else {
        std::vector<IsoDeposit> deps(nDeps);
        if (readFromCandidateView || theExtractForCandidate)
          deps = theExtractor->deposits(event, eventSetup, (*cands)[i]);
        else
          deps = theExtractor->deposits(event, eventSetup, muRef);
        for (unsigned int iDep = 0; iDep < nDeps; ++iDep) {
          deps2D[iDep][i] = deps[iDep];
        }
      }
    }  //! end for (nMuons)

    //! now fill in selectively
    for (unsigned int iDep = 0; iDep < nDeps; ++iDep) {
      //!some debugging stuff
      for (unsigned int iMu = 0; iMu < nMuons; ++iMu) {
        LogTrace(metname) << "Contents of " << theDepositNames[iDep] << " for a muon at index " << iMu;
        LogTrace(metname) << deps2D[iDep][iMu].print();
      }

      //! fill the maps here
      reco::IsoDepositMap::Filler filler(*depMaps[iDep]);

      //!now figure out the source handle (see getByToken above)
      if (readFromRecoMuon) {
        filler.insert(muons, deps2D[iDep].begin(), deps2D[iDep].end());
      } else if (readFromRecoTrack) {
        filler.insert(tracks, deps2D[iDep].begin(), deps2D[iDep].end());
      } else if (readFromCandidateView) {
        filler.insert(cands, deps2D[iDep].begin(), deps2D[iDep].end());
      } else {
        edm::LogError(metname) << "Inconsistent configuration: unknown type requested";
      }

      //! now actually fill
      filler.fill();
    }
  }  //! end if (nMuons>0)

  for (unsigned int iMap = 0; iMap < nDeps; ++iMap) {
    LogTrace(metname) << "About to put a deposit named " << theDepositNames[iMap] << " of size "
                      << depMaps[iMap]->size() << " into edm::Event";
    event.put(std::move(depMaps[iMap]), theDepositNames[iMap]);
  }

  LogTrace(metname) << " END OF EVENT "
                    << "================================";
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(MuIsoDepositProducer);

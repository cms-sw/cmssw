#include "RecoMuon/MuonIsolationProducers/src/MuIsoDepositProducer.h"

// Framework
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/MuonReco/interface/MuIsoDeposit.h"
#include "DataFormats/MuonReco/interface/MuIsoDepositFwd.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"


#include "RecoMuon/MuonIsolation/interface/Range.h"
#include "DataFormats/MuonReco/interface/Direction.h"

#include "RecoMuon/MuonIsolation/interface/MuIsoExtractor.h"
#include "RecoMuon/MuonIsolation/interface/MuIsoExtractorFactory.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <string>

using namespace edm;
using namespace std;
using namespace reco;
using namespace muonisolation;


/// constructor with config
MuIsoDepositProducer::MuIsoDepositProducer(const ParameterSet& par) :
  theConfig(par),
  theReadFromRecoMuon(par.getParameter<bool>("ReadFromRecoMuonCollection")),
  theMuonTrackRefType(par.getParameter<std::string>("MuonTrackRefType")),
  theMuonCollectionTag(par.getParameter<edm::InputTag>("inputMuonCollection")),
  theDepositNames(std::vector<std::string>(1,"")),
  theMultipleDepositsFlag(par.getParameter<bool>("MultipleDepositsFlag")),
  theExtractor(0)
  {
  LogDebug("RecoMuon|MuonIsolation")<<" MuIsoDepositProducer CTOR";
  if (! theMultipleDepositsFlag) produces<MuIsoDepositAssociationMap>();
  else {
    theDepositNames = par.getParameter<edm::ParameterSet>("ExtractorPSet")
      .getParameter<std::vector<std::string> >("DepositInstanceLabels");
    for (uint iDep=0; iDep<theDepositNames.size(); ++iDep){
      produces<MuIsoDepositAssociationMap>(theDepositNames[iDep]);
    }
  }
}

/// destructor
MuIsoDepositProducer::~MuIsoDepositProducer(){
  LogDebug("RecoMuon/MuIsoDepositProducer")<<" MuIsoDepositProducer DTOR";
  delete theExtractor;
}

/// build deposits
void MuIsoDepositProducer::produce(Event& event, const EventSetup& eventSetup){
  std::string metname = "RecoMuon|MuonIsolationProducers|MuIsoDepositProducer";

  LogDebug(metname)<<" Muon Deposit producing..."
		   <<" BEGINING OF EVENT " <<"================================";

  if (!theExtractor) {
    edm::ParameterSet extractorPSet = theConfig.getParameter<edm::ParameterSet>("ExtractorPSet");
    std::string extractorName = extractorPSet.getParameter<std::string>("ComponentName");
    theExtractor = MuIsoExtractorFactory::get()->create( extractorName, extractorPSet);
    LogDebug(metname)<<" Load extractor..."<<extractorName;
  }


  uint nDeps = theMultipleDepositsFlag ? theDepositNames.size() : 1;

  static const uint MAX_DEPS=10;
  std::auto_ptr<MuIsoDepositAssociationMap> depMaps[MAX_DEPS];
  for (uint i =0;i<nDeps; ++i){
    depMaps[i] =  std::auto_ptr<MuIsoDepositAssociationMap>(new MuIsoDepositAssociationMap());
  }


  // Take the muon container
  LogTrace(metname)<<" Taking the muons: "<<theMuonCollectionTag;
  Handle<TrackCollection> muonTracks;
  Handle<MuonCollection> muons;

  uint nMuons = 0;

  if (theReadFromRecoMuon){
    event.getByLabel(theMuonCollectionTag,muons);
    nMuons = muons->size();
    LogDebug(metname) <<"Got Muons of size "<<nMuons;
  } else {
    event.getByLabel(theMuonCollectionTag,muonTracks);
    nMuons = muonTracks->size();
    LogDebug(metname) <<"Got MuonTracks of size "<<nMuons;
  }


  for (uint i=0; i<  nMuons; ++i) {
    TrackRef mu;
    if (theReadFromRecoMuon){
      if (theMuonTrackRefType == "track"){
	mu = (*muons)[i].track();
      } else if (theMuonTrackRefType == "standAloneMuon"){
	mu = (*muons)[i].standAloneMuon();
      } else if (theMuonTrackRefType == "combinedMuon"){
	mu = (*muons)[i].combinedMuon();
      } else {
	edm::LogWarning(metname)<<"Wrong track type is supplied: breaking";
	break;
      }
    } else {
      mu = TrackRef(muonTracks, i);
    }
    if (! theMultipleDepositsFlag){
      MuIsoDeposit dep = theExtractor->deposit(event, eventSetup, *mu);
      LogTrace(metname)<<dep.print();
      depMaps[0]->insert(mu, dep);
    } else {
      std::vector<MuIsoDeposit> deps = theExtractor->deposits(event, eventSetup, *mu);
      for (uint iDep=0; iDep < nDeps; ++iDep){
	LogTrace(metname)<<deps[iDep].print();
	depMaps[iDep]->insert(mu, deps[iDep]);
      }
    }
  }


  for (uint iMap = 0; iMap < nDeps; ++iMap) event.put(depMaps[iMap], theDepositNames[iMap]);
  LogTrace(metname) <<" END OF EVENT " <<"================================";
}

#include "RecoMuon/MuonIsolationProducers/src/MuIsoDepositProducer.h"

// Framework
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/MuonReco/interface/MuIsoDeposit.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

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

  // Take the SA container
  LogTrace(metname)<<" Taking the muons: "<<theMuonCollectionTag;
  Handle<TrackCollection> muons;
  event.getByLabel(theMuonCollectionTag,muons);
  LogDebug(metname) <<"Got Muons of size "<<muons->size();


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


  for (uint i=0; i<muons->size(); i++) {
    TrackRef mu(muons,i);
    if (! theMultipleDepositsFlag){
      MuIsoDeposit dep = theExtractor->deposit(event, eventSetup, *mu);
      LogTrace(metname)<<dep.print();
      depMaps[0]->insert(mu, dep);
    } else {
      std::vector<MuIsoDeposit> deps = theExtractor->deposits(event, eventSetup, *mu);
      for (uint iDep=0; iDep < nDeps; iDep++){
	LogTrace(metname)<<deps[iDep].print();
	depMaps[iDep]->insert(mu, deps[iDep]);
      }
    }
  }


  for (uint iMap = 0; iMap < nDeps; ++iMap) event.put(depMaps[iMap], theDepositNames[iMap]);
  LogTrace(metname) <<" END OF EVENT " <<"================================";
}

#include "RecoMuon/L3MuonIsolationProducer/src/L3MuonIsolationProducer.h"

// Framework
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Handle.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/HLTReco/interface/HLTFilterObject.h"

#include "DataFormats/MuonReco/interface/MuIsoDeposit.h"

#include "RecoMuon/MuonIsolation/interface/Range.h"
#include "RecoMuon/MuonIsolation/interface/Direction.h"
#include "RecoMuon/MuonIsolation/src/TrackSelector.h"

#include "RecoMuon/MuonIsolation/interface/MuIsoExtractor.h"
#include "RecoMuon/MuonIsolation/src/TrackExtractor.h"

#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include <string>

using namespace edm;
using namespace std;
using namespace reco;
using namespace muonisolation;

/// constructor with config
L3MuonIsolationProducer::L3MuonIsolationProducer(const ParameterSet& par) :
  theMuonCollectionLabel(par.getUntrackedParameter<string>("MuonCollectionLabel"))
, theCuts( par.getParameter<std::vector<double> > ("EtaBounds"), 
             par.getParameter<std::vector<double> > ("ConeSizes"),
             par.getParameter<std::vector<double> > ("Thresholds") ),
  optOutputIsoDeposits(par.getParameter<bool>("OutputMuIsoDeposits"))
  {
  LogDebug("RecoMuon/L3MuonIsolationProducer")<<" L3MuonIsolationProducer CTOR";

  ParameterSet theMuIsoExtractorPSet = par.getParameter<ParameterSet>("MuIsoExtractorParameters");
  theTrackExtractor = muonisolation::TrackExtractor(theMuIsoExtractorPSet);

  if (optOutputIsoDeposits) produces<MuIsoDepositAssociationMap>();
  produces<MuIsoAssociationMap>();
}
  
/// destructor
L3MuonIsolationProducer::~L3MuonIsolationProducer(){
  LogDebug("RecoMuon/L3MuonIsolationProducer")<<" L3MuonIsolationProducer DTOR";
}

/// build deposits
void L3MuonIsolationProducer::produce(Event& event, const EventSetup& eventSetup){
  std::string metname = "RecoMuon/L3MuonIsolationProducer";
  
  LogDebug(metname)<<" L3 Muon Isolation producing..."
                    <<" BEGINING OF EVENT " <<"================================";

  // Take the SA container
  LogTrace(metname)<<" Taking the muons: "<<theMuonCollectionLabel;
  Handle<TrackCollection> muons;
  event.getByLabel(theMuonCollectionLabel,muons);

  std::auto_ptr<MuIsoDepositAssociationMap> depMap( new MuIsoDepositAssociationMap());
  std::auto_ptr<MuIsoAssociationMap> isoMap( new MuIsoAssociationMap());

  theTrackExtractor.fillVetos(event, eventSetup,*muons);

  for (unsigned int i=0; i<muons->size(); i++) {
    TrackRef mu(muons,i);

    MuIsoDeposit dep = theTrackExtractor.deposit(event, eventSetup, *mu);
    depMap->insert(mu, dep);

    const Cuts::CutSpec & cut = theCuts( mu->eta());
    double value = dep.depositWithin(cut.conesize);
    bool result = (value < cut.threshold); 
    LogTrace(metname)<<"deposit in cone: "<<value<<" is isolated: "<<result;
    isoMap->insert(mu, result);
  }

  if (optOutputIsoDeposits) event.put(depMap);
  event.put(isoMap);

  LogTrace(metname) <<" END OF EVENT " <<"================================";
}

#include "RecoMuon/L3MuonIsolationProducer/interface/L3MuonIsolationProducer.h"

// Framework
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Handle.h"

//#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
//#include "FWCore/Framework/interface/Frameworkfwd.h"
//#include "FWCore/ServiceRegistry/interface/Service.h"


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
L3MuonIsolationProducer::L3MuonIsolationProducer(const ParameterSet& parameterSet)
  :
    theCuts( parameterSet.getParameter<std::vector<double> > ("EtaBounds"), 
             parameterSet.getParameter<std::vector<double> > ("ConeSizes"),
             parameterSet.getParameter<std::vector<double> > ("Thresholds") )
  {
  LogDebug("RecoMuon/L3MuonIsolationProducer")<<" L3MuonIsolationProducer CTOR";

  theMuonCollectionLabel = parameterSet.getUntrackedParameter<string>("MuonCollectionLabel");
  theTrackCollectionLabel = parameterSet.getUntrackedParameter<string>("TrackCollectionLabel");

  theDiff_r = parameterSet.getParameter<double>("diff_r");
  theDiff_z = parameterSet.getParameter<double>("diff_z");
  theDR_Match = parameterSet.getParameter<double>("dR_Match");
  theDR_Veto  = parameterSet.getParameter<double>("dR_Veto");
  theDR_Max   = parameterSet.getParameter<double>("dR_Max");  

  theDepositLabel = parameterSet.getUntrackedParameter<string>("DepositLabel");

  produces<MuIsoDepositCollection>();
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

//  std::string extractorName = "muonisolation::TrackExtractor";
//  edm::ESHandle<MuIsoExtractor> extractorESH;
//  eventSetup.get<TrackingComponentsRecord>().get( extractorName, extractorESH);
//  const MuIsoExtractor * theExtractor = extractorESH.product();
  muonisolation::TrackExtractor extractor( theDiff_r, theDiff_z, theDR_Match, theDR_Veto,
      theTrackCollectionLabel, theDepositLabel);

  std::auto_ptr<MuIsoDepositCollection> depCollection( new MuIsoDepositCollection());
  std::auto_ptr<MuIsoAssociationMap> depMap( new MuIsoAssociationMap());

  vector<Direction> vetoDirections;
  for (unsigned int i=0; i<muons->size(); i++) {
    TrackRef mu(muons,i);
    vetoDirections.push_back( Direction(mu->eta(), mu->phi0()) );
  }
 
  for (unsigned int i=0; i<muons->size(); i++) {
    TrackRef mu(muons,i);

    const Cuts::CutSpec & cut = theCuts( mu->eta() );

    vector<MuIsoDeposit> deposits = extractor.deposits(event, *mu, vetoDirections, cut.conesize);
    
    const MuIsoDeposit & deposit = deposits[0]; //FIXME check size and thrown exception eventually

    depCollection->push_back(deposit);

    double value = deposit.depositWithin(cut.conesize);
    bool result = (value < cut.threshold); 
    LogTrace(metname)<<"deposit in cone: "<<value<<" is isolated: "<<result;
    depMap->insert(mu, result);
  }

  LogTrace(metname) << " deposite Collections: " << depCollection->size();
  event.put(depCollection);
  event.put(depMap);

  LogTrace(metname) <<" END OF EVENT " <<"================================";
}

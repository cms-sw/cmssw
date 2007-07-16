#include "L3MuonIsolationProducer.h"

// Framework
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "DataFormats/MuonReco/interface/MuIsoDeposit.h"

#include "RecoMuon/MuonIsolation/interface/Range.h"
#include "DataFormats/MuonReco/interface/Direction.h"

#include "RecoMuon/MuonIsolation/interface/MuIsoExtractor.h"
#include "RecoMuon/MuonIsolation/interface/MuIsoExtractorFactory.h"

#include "L3NominalEfficiencyConfigurator.h"

#include <string>

using namespace edm;
using namespace std;
using namespace reco;
using namespace muonisolation;

/// constructor with config
L3MuonIsolationProducer::L3MuonIsolationProducer(const ParameterSet& par) :
  theConfig(par),
  theMuonCollectionLabel(par.getParameter<InputTag>("inputMuonCollection")),
  optOutputIsoDeposits(par.getParameter<bool>("OutputMuIsoDeposits")),
  theExtractor(0)
  {
  LogDebug("RecoMuon|L3MuonIsolationProducer")<<" L3MuonIsolationProducer CTOR";

  if (optOutputIsoDeposits) produces<MuIsoDepositAssociationMap>();
  produces<MuIsoAssociationMap>();
}
  
/// destructor
L3MuonIsolationProducer::~L3MuonIsolationProducer(){
  LogDebug("RecoMuon|L3MuonIsolationProducer")<<" L3MuonIsolationProducer DTOR";
  if (theExtractor) delete theExtractor;
}

void L3MuonIsolationProducer::beginJob(const edm::EventSetup& iSetup)
{

  //
  // Extractor
  //
  edm::ParameterSet extractorPSet = theConfig.getParameter<edm::ParameterSet>("ExtractorPSet");
  std::string extractorName = extractorPSet.getParameter<std::string>("ComponentName");
  theExtractor = MuIsoExtractorFactory::get()->create( extractorName, extractorPSet);
  std::string depositType = extractorPSet.getUntrackedParameter<std::string>("DepositLabel");
  
  //
  // Cuts
  //
  edm::ParameterSet cutsPSet = theConfig.getParameter<edm::ParameterSet>("CutsPSet");
  std::string cutsName = cutsPSet.getParameter<std::string>("ComponentName");
  if (cutsName == "SimpleCuts") {
    theCuts = Cuts(cutsPSet);
  } 
  else if (
//        (cutsName== "L3NominalEfficiencyCuts_PXLS" && depositType=="PXLS")
//     || (cutsName== "L3NominalEfficiencyCuts_TRKS" && depositType=="TRKS") 
//! test cutsName only. The depositType is informational only (has not been used so far) [VK]
	   (cutsName== "L3NominalEfficiencyCuts_PXLS" )
	   || (cutsName== "L3NominalEfficiencyCuts_TRKS") ) {
    theCuts = L3NominalEfficiencyConfigurator(cutsPSet).cuts();
  } 
  else {
    LogError("L3MuonIsolationProducer::beginJob")
      <<"cutsName: "<<cutsPSet<<" is not recognized:"
      <<" theCuts not set!";
  }
  LogTrace("")<< theCuts.print();
}

void L3MuonIsolationProducer::produce(Event& event, const EventSetup& eventSetup){
  std::string metname = "RecoMuon|L3MuonIsolationProducer";
  
  LogDebug(metname)<<" L3 Muon Isolation producing..."
                    <<" BEGINING OF EVENT " <<"================================";

  // Take the SA container
  LogTrace(metname)<<" Taking the muons: "<<theMuonCollectionLabel;
  Handle<TrackCollection> muons;
  event.getByLabel(theMuonCollectionLabel,muons);

  std::auto_ptr<MuIsoDepositAssociationMap> depMap( new MuIsoDepositAssociationMap());
  std::auto_ptr<MuIsoAssociationMap> isoMap( new MuIsoAssociationMap());


  //
  // get Vetos and deposits
  //
  MuIsoDeposit::Vetos vetos;
  typedef std::vector< std::pair<TrackRef,MuIsoDeposit> > MuonsWithDeposits;
  MuonsWithDeposits muonsWithDeposits;

  for (unsigned int i=0; i<muons->size(); i++) {
    TrackRef mu(muons,i);
    MuIsoDeposit dep = theExtractor->deposit(event, eventSetup, *mu);
    vetos.push_back(dep.veto());
    muonsWithDeposits.push_back( std::make_pair(mu,dep) ); 
  }

  //
  // add here additional vetos
  //
  //.....

  //
  // actual cut step
  //
  for (MuonsWithDeposits::const_iterator imd = muonsWithDeposits.begin(),
       imdEnd = muonsWithDeposits.end(); imd != imdEnd; ++imd) {
    const TrackRef & mu = imd->first;
    const MuIsoDeposit & deposit = imd->second;
    LogTrace(metname)<< deposit.print();
    const Cuts::CutSpec & cut = theCuts( mu->eta());
    double value = deposit.depositWithin(cut.conesize, vetos);
    bool result = (value < cut.threshold); 
    LogTrace(metname)<<"deposit in cone: "<<value<<" is isolated: "<<result;
    if (optOutputIsoDeposits) depMap->insert(mu, deposit);
    isoMap->insert(mu, result);
  }

  //
  // store
  //
  if (optOutputIsoDeposits) event.put(depMap);
  event.put(isoMap);

  LogTrace(metname) <<" END OF EVENT " <<"================================";
}

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
#include "DataFormats/MuonReco/interface/MuIsoDepositFwd.h"

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
  theExtractor(0),
  theTrackPt_Min(-1)
  {
  LogDebug("RecoMuon|L3MuonIsolationProducer")<<" L3MuonIsolationProducer CTOR";

  if (optOutputIsoDeposits) produces<reco::IsoDepositMap>();
  produces<reco::MuIsoFlagMap>();
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
  //! get min pt for the track to go into sumPt
  theTrackPt_Min = theConfig.getParameter<double>("TrackPt_Min");
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

  std::auto_ptr<reco::IsoDepositMap> depMap( new reco::IsoDepositMap());
  std::auto_ptr<reco::MuIsoFlagMap> isoMap( new reco::MuIsoFlagMap());


  //
  // get Vetos and deposits
  //
  uint nMuons = muons->size();

  MuIsoDeposit::Vetos vetos(nMuons);
  
  std::vector<MuIsoDeposit> deps(nMuons);
  std::vector<bool> isos(nMuons, false);

  for (unsigned int i=0; i<nMuons; i++) {
    TrackRef mu(muons,i);
    deps[i] = theExtractor->deposit(event, eventSetup, *mu);
    vetos[i] = deps[i].veto();
  }

  //
  // add here additional vetos
  //
  //.....

  //
  // actual cut step
  //
  for(uint iMu=0; iMu < nMuons; ++iMu){
    const reco::Track* mu = &(*muons)[iMu];

    const MuIsoDeposit & deposit = deps[iMu];
    LogTrace(metname)<< deposit.print();

    const Cuts::CutSpec & cut = theCuts( mu->eta());
    std::pair<double, int> sumAndCount = deposit.depositAndCountWithin(cut.conesize, vetos, theTrackPt_Min);

    double value = sumAndCount.first;
    bool result = (value < cut.threshold); 
    LogTrace(metname)<<"deposit in cone: "<<value<<" is isolated: "<<result;

    isos[iMu] = result;
  }

  //
  // store
  //
  if (optOutputIsoDeposits){
    reco::IsoDepositMap::Filler depFiller(*depMap);
    depFiller.insert(muons, deps.begin(), deps.end());
    depFiller.fill();
    event.put(depMap);
  }
  reco::MuIsoFlagMap::Filler isoFiller(*isoMap);
  isoFiller.insert(muons, isos.begin(), isos.end());
  isoFiller.fill();
  event.put(isoMap);

  LogTrace(metname) <<" END OF EVENT " <<"================================";
}

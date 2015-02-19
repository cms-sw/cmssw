#include "L3MuonIsolationProducer.h"

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositFwd.h"

#include "RecoMuon/MuonIsolation/interface/Range.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositDirection.h"

#include "PhysicsTools/IsolationAlgos/interface/IsoDepositExtractor.h"
#include "PhysicsTools/IsolationAlgos/interface/IsoDepositExtractorFactory.h"

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

  theMuonCollectionToken = consumes<RecoChargedCandidateCollection>(theMuonCollectionLabel);

  if (optOutputIsoDeposits) produces<reco::IsoDepositMap>();
  produces<edm::ValueMap<bool> >();

  //
  // Extractor
  //
  edm::ParameterSet extractorPSet = theConfig.getParameter<edm::ParameterSet>("ExtractorPSet");
  //! get min pt for the track to go into sumPt
  theTrackPt_Min = theConfig.getParameter<double>("TrackPt_Min");
  std::string extractorName = extractorPSet.getParameter<std::string>("ComponentName");
  theExtractor = IsoDepositExtractorFactory::get()->create( extractorName, extractorPSet, consumesCollector());
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

  // (kludge) additional cut on the number of tracks
  theMaxNTracks = cutsPSet.getParameter<int>("maxNTracks");
  theApplyCutsORmaxNTracks = cutsPSet.getParameter<bool>("applyCutsORmaxNTracks");
}

/// destructor
L3MuonIsolationProducer::~L3MuonIsolationProducer(){
  LogDebug("RecoMuon|L3MuonIsolationProducer")<<" L3MuonIsolationProducer DTOR";
  if (theExtractor) delete theExtractor;
}

void L3MuonIsolationProducer::produce(Event& event, const EventSetup& eventSetup){
  std::string metname = "RecoMuon|L3MuonIsolationProducer";

  LogDebug(metname)<<" L3 Muon Isolation producing..."
                    <<" BEGINING OF EVENT " <<"================================";

  // Take the SA container
  LogTrace(metname)<<" Taking the muons: "<<theMuonCollectionLabel;
  Handle<TrackCollection> muons;
  event.getByToken(theMuonCollectionToken,muons);

  std::auto_ptr<reco::IsoDepositMap> depMap( new reco::IsoDepositMap());
  std::auto_ptr<edm::ValueMap<bool> > isoMap( new edm::ValueMap<bool> ());


  //
  // get Vetos and deposits
  //
  unsigned int nMuons = muons->size();

  IsoDeposit::Vetos vetos(nMuons);

  std::vector<IsoDeposit> deps(nMuons);
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
  for(unsigned int iMu=0; iMu < nMuons; ++iMu){
    const reco::Track* mu = &(*muons)[iMu];

    const IsoDeposit & deposit = deps[iMu];
    LogTrace(metname)<< deposit.print();

    const Cuts::CutSpec & cut = theCuts( mu->eta());
    std::pair<double, int> sumAndCount = deposit.depositAndCountWithin(cut.conesize, vetos, theTrackPt_Min);

    double value = sumAndCount.first;
    int count = sumAndCount.second;

    bool result = (value < cut.threshold);
    if (theApplyCutsORmaxNTracks ) result |= count <= theMaxNTracks;
    LogTrace(metname)<<"deposit in cone: "<<value<<"with count "<<count<<" is isolated: "<<result;

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
  edm::ValueMap<bool> ::Filler isoFiller(*isoMap);
  isoFiller.insert(muons, isos.begin(), isos.end());
  isoFiller.fill();
  event.put(isoMap);

  LogTrace(metname) <<" END OF EVENT " <<"================================";
}

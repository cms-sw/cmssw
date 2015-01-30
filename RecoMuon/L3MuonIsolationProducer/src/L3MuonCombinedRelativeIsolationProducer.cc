#include "L3MuonCombinedRelativeIsolationProducer.h"

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
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"

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
L3MuonCombinedRelativeIsolationProducer::L3MuonCombinedRelativeIsolationProducer(const ParameterSet& par) :
  theConfig(par),
  theMuonCollectionLabel(par.getParameter<InputTag>("inputMuonCollection")),
  optOutputIsoDeposits(par.getParameter<bool>("OutputMuIsoDeposits")),
  useCaloIso(par.existsAs<bool>("UseCaloIso") ?
	     par.getParameter<bool>("UseCaloIso") : true),
  useRhoCorrectedCaloDeps(par.existsAs<bool>("UseRhoCorrectedCaloDeposits") ?
			  par.getParameter<bool>("UseRhoCorrectedCaloDeposits") : false),
  theCaloDepsLabel(par.existsAs<InputTag>("CaloDepositsLabel") ?
                   par.getParameter<InputTag>("CaloDepositsLabel") :
                   InputTag("hltL3CaloMuonCorrectedIsolations")),
  caloExtractor(0),
  trkExtractor(0),
  theTrackPt_Min(-1),
  printDebug (par.getParameter<bool>("printDebug"))
  {
  LogDebug("RecoMuon|L3MuonCombinedRelativeIsolationProducer")<<" L3MuonCombinedRelativeIsolationProducer CTOR";

  theMuonCollectionToken = consumes<RecoChargedCandidateCollection>(theMuonCollectionLabel);
  theCaloDepsToken = consumes<edm::ValueMap<float> >(theCaloDepsLabel);

  if (optOutputIsoDeposits) {
    produces<reco::IsoDepositMap>("trkIsoDeposits");
    if( useRhoCorrectedCaloDeps==false ) // otherwise, calo deposits have been previously computed
      produces<reco::IsoDepositMap>("caloIsoDeposits");
    //produces<std::vector<double> >("combinedRelativeIsoDeposits");
    produces<edm::ValueMap<double> >("combinedRelativeIsoDeposits");
  }
  produces<edm::ValueMap<bool> >();

  //
  // Extractor
  //
  // Calorimeters (ONLY if not previously computed)
  //
  if( useCaloIso && (useRhoCorrectedCaloDeps==false) ) {
    edm::ParameterSet caloExtractorPSet = theConfig.getParameter<edm::ParameterSet>("CaloExtractorPSet");

    std::string caloExtractorName = caloExtractorPSet.getParameter<std::string>("ComponentName");
    caloExtractor = IsoDepositExtractorFactory::get()->create( caloExtractorName, caloExtractorPSet, consumesCollector());
    //std::string caloDepositType = caloExtractorPSet.getUntrackedParameter<std::string>("DepositLabel"); // N.B. Not used in the following!
  }

  // Tracker
  //
  edm::ParameterSet trkExtractorPSet = theConfig.getParameter<edm::ParameterSet>("TrkExtractorPSet");

  theTrackPt_Min = theConfig.getParameter<double>("TrackPt_Min");
  std::string trkExtractorName = trkExtractorPSet.getParameter<std::string>("ComponentName");
  trkExtractor = IsoDepositExtractorFactory::get()->create( trkExtractorName, trkExtractorPSet, consumesCollector());
  //std::string trkDepositType = trkExtractorPSet.getUntrackedParameter<std::string>("DepositLabel"); // N.B. Not used in the following!

  //
  // Cuts for track isolation
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
    LogError("L3MuonCombinedRelativeIsolationProducer::beginJob")
      <<"cutsName: "<<cutsPSet<<" is not recognized:"
      <<" theCuts not set!";
  }
  LogTrace("")<< theCuts.print();

  // (kludge) additional cut on the number of tracks
  theMaxNTracks = cutsPSet.getParameter<int>("maxNTracks");
  theApplyCutsORmaxNTracks = cutsPSet.getParameter<bool>("applyCutsORmaxNTracks");

}

/// destructor
L3MuonCombinedRelativeIsolationProducer::~L3MuonCombinedRelativeIsolationProducer(){
  LogDebug("RecoMuon|L3MuonCombinedRelativeIsolationProducer")<<" L3MuonCombinedRelativeIsolationProducer DTOR";
  if (caloExtractor) delete caloExtractor;
  if (trkExtractor) delete trkExtractor;
}

/// ParameterSet descriptions
void L3MuonCombinedRelativeIsolationProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("UseRhoCorrectedCaloDeposits",false);
  desc.add<bool>("UseCaloIso",true);
  desc.add<edm::InputTag>("CaloDepositsLabel",edm::InputTag("hltL3CaloMuonCorrectedIsolations"));
  desc.add<edm::InputTag>("inputMuonCollection",edm::InputTag("hltL3MuonCandidates"));
  desc.add<bool>("OutputMuIsoDeposits",true);
  desc.add<double>("TrackPt_Min",-1.0);
  desc.add<bool>("printDebug",false);
  edm::ParameterSetDescription cutsPSet;
  {
    cutsPSet.add<std::vector<double> >("ConeSizes",std::vector<double>(1, 0.24));
    cutsPSet.add<std::string>("ComponentName","SimpleCuts");
    cutsPSet.add<std::vector<double> >("Thresholds",std::vector<double>(1, 0.1));
    cutsPSet.add<int>("maxNTracks",-1);
    cutsPSet.add<std::vector<double> >("EtaBounds",std::vector<double>(1, 2.411));
    cutsPSet.add<bool>("applyCutsORmaxNTracks",false);
  }
  desc.add<edm::ParameterSetDescription>("CutsPSet",cutsPSet);
  edm::ParameterSetDescription trkExtractorPSet;
  {
    trkExtractorPSet.add<double>("Chi2Prob_Min", -1.0);
    trkExtractorPSet.add<double>("Chi2Ndof_Max", 1.0E64);
    trkExtractorPSet.add<double>("Diff_z",0.2);
    trkExtractorPSet.add<edm::InputTag>("inputTrackCollection",edm::InputTag("hltPixelTracks"));
    trkExtractorPSet.add<double>("ReferenceRadius",6.0);
    trkExtractorPSet.add<edm::InputTag>("BeamSpotLabel",edm::InputTag("hltOnlineBeamSpot"));
    trkExtractorPSet.add<std::string>("ComponentName","PixelTrackExtractor");
    trkExtractorPSet.add<double>("DR_Max",0.24);
    trkExtractorPSet.add<double>("Diff_r",0.1);
    trkExtractorPSet.add<bool>("VetoLeadingTrack",true);
    trkExtractorPSet.add<double>("DR_VetoPt",0.025);
    trkExtractorPSet.add<double>("DR_Veto",0.01);
    trkExtractorPSet.add<unsigned int>("NHits_Min",0);
    trkExtractorPSet.add<double>("Pt_Min",-1.0);
    trkExtractorPSet.addUntracked<std::string>("DepositLabel","PXLS");
    trkExtractorPSet.add<std::string>("BeamlineOption","BeamSpotFromEvent");
    trkExtractorPSet.add<bool>("PropagateTracksToRadius",true);
    trkExtractorPSet.add<double>("PtVeto_Min",2.0);
  }
  desc.add<edm::ParameterSetDescription>("TrkExtractorPSet",trkExtractorPSet);
  edm::ParameterSetDescription caloExtractorPSet;
  {
    caloExtractorPSet.add<double>("DR_Veto_H",0.1);
    caloExtractorPSet.add<bool>("Vertex_Constraint_Z",false);
    caloExtractorPSet.add<double>("Threshold_H",0.5);
    caloExtractorPSet.add<std::string>("ComponentName","CaloExtractor");
    caloExtractorPSet.add<double>("Threshold_E",0.2);
    caloExtractorPSet.add<double>("DR_Max",0.24);
    caloExtractorPSet.add<double>("DR_Veto_E",0.07);
    caloExtractorPSet.add<double>("Weight_E",1.5);
    caloExtractorPSet.add<bool>("Vertex_Constraint_XY",false);
    caloExtractorPSet.addUntracked<std::string>("DepositLabel","EcalPlusHcal");
    caloExtractorPSet.add<edm::InputTag>("CaloTowerCollectionLabel",edm::InputTag("hltTowerMakerForMuons"));
    caloExtractorPSet.add<double>("Weight_H",1.0);
  }
  desc.add<edm::ParameterSetDescription>("CaloExtractorPSet",caloExtractorPSet);
  descriptions.add("hltL3MuonIsolations", desc);
}


void L3MuonCombinedRelativeIsolationProducer::produce(Event& event, const EventSetup& eventSetup){
  std::string metname = "RecoMuon|L3MuonCombinedRelativeIsolationProducer";

  if (printDebug) std::cout  <<" L3 Muon Isolation producing..."
            <<" BEGINING OF EVENT " <<"================================" <<std::endl;

  // Take the SA container
  if (printDebug) std::cout  <<" Taking the muons: "<<theMuonCollectionLabel << std::endl;
  Handle<RecoChargedCandidateCollection> muons;
  event.getByToken(theMuonCollectionToken,muons);

  // Take calo deposits with rho corrections (ONLY if previously computed)
  Handle< edm::ValueMap<float> > caloDepWithCorrMap;
  if( useRhoCorrectedCaloDeps )
    event.getByToken(theCaloDepsToken, caloDepWithCorrMap);

  std::auto_ptr<reco::IsoDepositMap> caloDepMap( new reco::IsoDepositMap());
  std::auto_ptr<reco::IsoDepositMap> trkDepMap( new reco::IsoDepositMap());

  std::auto_ptr<edm::ValueMap<bool> > comboIsoDepMap( new edm::ValueMap<bool> ());

  //std::auto_ptr<std::vector<double> > combinedRelativeDeps(new std::vector<double>());
  std::auto_ptr<edm::ValueMap<double> > combinedRelativeDepMap(new edm::ValueMap<double>());


  //
  // get Vetos and deposits
  //
  unsigned int nMuons = muons->size();

  IsoDeposit::Vetos trkVetos(nMuons);
  std::vector<IsoDeposit> trkDeps(nMuons);


  // IsoDeposit::Vetos caloVetos(nMuons);
  // std::vector<IsoDeposit> caloDeps(nMuons);
  // std::vector<float> caloCorrDeps(nMuons, 0.);  // if calo deposits with corrections available

  IsoDeposit::Vetos caloVetos;
  std::vector<IsoDeposit> caloDeps;
  std::vector<float> caloCorrDeps;  // if calo deposits with corrections available

  if(useCaloIso && useRhoCorrectedCaloDeps) {
    caloCorrDeps.resize(nMuons, 0.);
  }
  else if (useCaloIso) {
    caloVetos.resize(nMuons);
    caloDeps.resize(nMuons);
  }

  std::vector<double> combinedRelativeDeps(nMuons, 0.);
  std::vector<bool> combinedRelativeIsos(nMuons, false);

  for (unsigned int i=0; i<nMuons; i++) {

    RecoChargedCandidateRef candref(muons,i);
    TrackRef mu = candref->track();

    trkDeps[i] = trkExtractor->deposit(event, eventSetup, *mu);
    trkVetos[i] = trkDeps[i].veto();

    if( useCaloIso && useRhoCorrectedCaloDeps ) {
      caloCorrDeps[i] = (*caloDepWithCorrMap)[candref];
    }
    else if (useCaloIso) {
      caloDeps[i] = caloExtractor->deposit(event, eventSetup, *mu);
      caloVetos[i] = caloDeps[i].veto();
    }

  }

  //
  // add here additional vetos
  //
  //.....

  //
  // actual cut step
  //

  if (printDebug) std::cout  << "Looping over deposits...." << std::endl;
  for(unsigned int iMu=0; iMu < nMuons; ++iMu){

    if (printDebug) std::cout  << "Muon number = " << iMu << std::endl;
    TrackRef mu = (*muons)[iMu].track();

    // cuts
    const Cuts::CutSpec & cut = theCuts( mu->eta());


    if (printDebug) std::cout << "CUTDEBUG: Muon eta = " << mu->eta() << std::endl
                              << "CUTDEBUG: Muon pt  = " <<  mu->pt() << std::endl
                              << "CUTDEBUG: minEta   = " << cut.etaRange.min() << std::endl
                              << "CUTDEBUG: maxEta   = " << cut.etaRange.max() << std::endl
                              << "CUTDEBUG: consize  = " << cut.conesize << std::endl
                              << "CUTDEBUG: thresho  = " << cut.threshold << std::endl;

    const IsoDeposit & trkDeposit = trkDeps[iMu];
    if (printDebug) std::cout  << trkDeposit.print();
    std::pair<double, int> trkIsoSumAndCount = trkDeposit.depositAndCountWithin(cut.conesize, trkVetos, theTrackPt_Min);

    double caloIsoSum = 0.;
    if( useCaloIso && useRhoCorrectedCaloDeps ) {
      caloIsoSum = caloCorrDeps[iMu];
      if(caloIsoSum<0.) caloIsoSum = 0.;
      if(printDebug) std::cout << "Rho-corrected calo deposit (min. 0) = " << caloIsoSum << std::endl;
    }
    else if (useCaloIso) {
      const IsoDeposit & caloDeposit = caloDeps[iMu];
      if (printDebug) std::cout  << caloDeposit.print();
      caloIsoSum = caloDeposit.depositWithin(cut.conesize, caloVetos);
    }

    double trkIsoSum = trkIsoSumAndCount.first;
    int count = trkIsoSumAndCount.second;

    double muPt = mu->pt();
    if( muPt<1. ) muPt = 1.;
    double combinedRelativeDeposit = ((trkIsoSum + caloIsoSum ) / muPt);
    bool result = ( combinedRelativeDeposit < cut.threshold);
    if (theApplyCutsORmaxNTracks ) result |= count <= theMaxNTracks;
    if (printDebug) std::cout  <<"  trk dep in cone:  " << trkIsoSum << "  with count "<<count <<std::endl
              <<"  calo dep in cone: " << caloIsoSum << std::endl
              <<"  muPt: " << muPt << std::endl
              <<"  relIso:  " <<combinedRelativeDeposit  << std::endl
              <<"  is isolated: "<<result << std::endl;

    combinedRelativeIsos[iMu] = result;
    //combinedRelativeDeps->push_back(combinedRelativeDeposit);
    combinedRelativeDeps[iMu] = combinedRelativeDeposit;
  }

  //
  // store
  //
  if (optOutputIsoDeposits){

    reco::IsoDepositMap::Filler depFillerTrk(*trkDepMap);
    depFillerTrk.insert(muons, trkDeps.begin(), trkDeps.end());
    depFillerTrk.fill();
    event.put(trkDepMap, "trkIsoDeposits");

    if( useCaloIso && (useRhoCorrectedCaloDeps==false) ) {
      reco::IsoDepositMap::Filler depFillerCalo(*caloDepMap);
      depFillerCalo.insert(muons, caloDeps.begin(), caloDeps.end());
      depFillerCalo.fill();
      event.put(caloDepMap, "caloIsoDeposits");
    }

    //event.put(combinedRelativeDeps, "combinedRelativeIsoDeposits");
    edm::ValueMap<double>::Filler depFillerCombRel(*combinedRelativeDepMap);
    depFillerCombRel.insert(muons, combinedRelativeDeps.begin(), combinedRelativeDeps.end());
    depFillerCombRel.fill();
    event.put(combinedRelativeDepMap, "combinedRelativeIsoDeposits");

  }
  edm::ValueMap<bool>::Filler isoFiller(*comboIsoDepMap);
  isoFiller.insert(muons, combinedRelativeIsos.begin(), combinedRelativeIsos.end());
  isoFiller.fill();
  event.put(comboIsoDepMap);

  if (printDebug) std::cout  <<" END OF EVENT " <<"================================";
}

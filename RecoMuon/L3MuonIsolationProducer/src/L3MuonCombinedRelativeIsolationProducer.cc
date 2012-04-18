#include "L3MuonCombinedRelativeIsolationProducer.h"

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
  useRhoCorrectedCaloDeps(par.getParameter<bool>("UseRhoCorrectedCaloDeposits")),
  theCaloDepsLabel(par.getParameter<InputTag>("CaloDepositsLabel")),
  caloExtractor(0),
  trkExtractor(0),
  theTrackPt_Min(-1),
  printDebug (par.getParameter<bool>("printDebug"))
  {
  LogDebug("RecoMuon|L3MuonCombinedRelativeIsolationProducer")<<" L3MuonCombinedRelativeIsolationProducer CTOR";

  if (optOutputIsoDeposits) {
    produces<reco::IsoDepositMap>("trkIsoDeposits");
    if( useRhoCorrectedCaloDeps==false ) // otherwise, calo deposits have been previously computed
      produces<reco::IsoDepositMap>("caloIsoDeposits");
    //produces<std::vector<double> >("combinedRelativeIsoDeposits");
    produces<edm::ValueMap<double> >("combinedRelativeIsoDeposits");
  }
  produces<edm::ValueMap<bool> >();

}
  
/// destructor
L3MuonCombinedRelativeIsolationProducer::~L3MuonCombinedRelativeIsolationProducer(){
  LogDebug("RecoMuon|L3MuonCombinedRelativeIsolationProducer")<<" L3MuonCombinedRelativeIsolationProducer DTOR";
  if (caloExtractor) delete caloExtractor;
  if (trkExtractor) delete trkExtractor;
}

void L3MuonCombinedRelativeIsolationProducer::beginJob()
{

  //
  // Extractor
  //
  // Calorimeters (ONLY if not previously computed)
  //
  if( useRhoCorrectedCaloDeps==false ) {
    edm::ParameterSet caloExtractorPSet = theConfig.getParameter<edm::ParameterSet>("CaloExtractorPSet");
  
    theTrackPt_Min = theConfig.getParameter<double>("TrackPt_Min");
    std::string caloExtractorName = caloExtractorPSet.getParameter<std::string>("ComponentName");
    caloExtractor = IsoDepositExtractorFactory::get()->create( caloExtractorName, caloExtractorPSet);
    //std::string caloDepositType = caloExtractorPSet.getUntrackedParameter<std::string>("DepositLabel"); // N.B. Not used in the following!
  }

  // Tracker
  //
  edm::ParameterSet trkExtractorPSet = theConfig.getParameter<edm::ParameterSet>("TrkExtractorPSet");

  std::string trkExtractorName = trkExtractorPSet.getParameter<std::string>("ComponentName");
  trkExtractor = IsoDepositExtractorFactory::get()->create( trkExtractorName, trkExtractorPSet);
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

void L3MuonCombinedRelativeIsolationProducer::produce(Event& event, const EventSetup& eventSetup){
  std::string metname = "RecoMuon|L3MuonCombinedRelativeIsolationProducer";
  
  if (printDebug) std::cout  <<" L3 Muon Isolation producing..."
            <<" BEGINING OF EVENT " <<"================================" <<std::endl;

  // Take the SA container
  if (printDebug) std::cout  <<" Taking the muons: "<<theMuonCollectionLabel << std::endl;
  Handle<TrackCollection> muons;
  event.getByLabel(theMuonCollectionLabel,muons);

  // Take calo deposits with rho corrections (ONLY if previously computed)
  Handle< edm::ValueMap<float> > caloDepWithCorrMap;
  if( useRhoCorrectedCaloDeps )
    event.getByLabel(theCaloDepsLabel, caloDepWithCorrMap);

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
  

  IsoDeposit::Vetos caloVetos(nMuons);  
  std::vector<IsoDeposit> caloDeps(nMuons);
  std::vector<float> caloCorrDeps(nMuons, 0.);  // if calo deposits with corrections available
  
  
  std::vector<double> combinedRelativeDeps(nMuons, 0.);

  std::vector<bool> combinedRelativeIsos(nMuons, false);

  


  for (unsigned int i=0; i<nMuons; i++) {
    
    TrackRef mu(muons,i);
    
    trkDeps[i] = trkExtractor->deposit(event, eventSetup, *mu);
    trkVetos[i] = trkDeps[i].veto();

    if( useRhoCorrectedCaloDeps ) {
      caloCorrDeps[i] = (*caloDepWithCorrMap)[mu];
    }
    else {
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
    const reco::Track* mu = &(*muons)[iMu];

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
    if( useRhoCorrectedCaloDeps ) {
      caloIsoSum = caloCorrDeps[iMu];
      if(caloIsoSum<0.) caloIsoSum = 0.;
      if(printDebug) std::cout << "Rho-corrected calo deposit (min. 0) = " << caloIsoSum << std::endl;
    }
    else {
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

    if( useRhoCorrectedCaloDeps==false ) {
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

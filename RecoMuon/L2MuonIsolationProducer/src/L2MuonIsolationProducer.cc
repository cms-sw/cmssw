/**  \class L2MuonIsolationProducer
 *
 *   \author  J. Alcaraz
 */

// Framework
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/RecoCandidate/interface/IsoDepositDirection.h"
#include "RecoMuon/L2MuonIsolationProducer/src/L2MuonIsolationProducer.h"

#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositFwd.h"

#include "PhysicsTools/IsolationAlgos/interface/IsoDepositExtractor.h"
#include "PhysicsTools/IsolationAlgos/interface/IsoDepositExtractorFactory.h"
#include "RecoMuon/MuonIsolation/interface/MuonIsolatorFactory.h"

#include <string>

using namespace edm;
using namespace std;
using namespace reco;
using namespace muonisolation;

/// constructor with config
L2MuonIsolationProducer::L2MuonIsolationProducer(const ParameterSet& par) :
  theSACollectionLabel(par.getParameter<edm::InputTag>("StandAloneCollectionLabel")),
  theExtractor(0),
  theDepositIsolator(0)
{
  LogDebug("Muon|RecoMuon|L2MuonIsolationProducer")<<" L2MuonIsolationProducer constructor called";

  theSACollectionToken = consumes<RecoChargedCandidateCollection>(theSACollectionLabel);

  //
  // Extractor
  //
  edm::ParameterSet extractorPSet = par.getParameter<edm::ParameterSet>("ExtractorPSet");
  std::string extractorName = extractorPSet.getParameter<std::string>("ComponentName");
  theExtractor = IsoDepositExtractorFactory::get()->create( extractorName, extractorPSet, consumesCollector());


  edm::ParameterSet isolatorPSet = par.getParameter<edm::ParameterSet>("IsolatorPSet");
  bool haveIsolator = !isolatorPSet.empty();
  optOutputDecision = haveIsolator;
  if (optOutputDecision){
    std::string type = isolatorPSet.getParameter<std::string>("ComponentName");
    theDepositIsolator = MuonIsolatorFactory::get()->create(type, isolatorPSet, consumesCollector());
  }
  if (optOutputDecision) produces<edm::ValueMap<bool> >();
  produces<reco::IsoDepositMap>();

  optOutputIsolatorFloat = par.getParameter<bool>("WriteIsolatorFloat");
  if (optOutputIsolatorFloat && haveIsolator){
    produces<edm::ValueMap<float> >();
  }
}

/// destructor
L2MuonIsolationProducer::~L2MuonIsolationProducer(){
  LogDebug("Muon|RecoMuon|L2MuonIsolationProducer")<<" L2MuonIsolationProducer destructor called";
  if (theExtractor) delete theExtractor;
}

/// ParameterSet descriptions
void L2MuonIsolationProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("StandAloneCollectionLabel",edm::InputTag("hltL2MuonCandidates"));
  edm::ParameterSetDescription extractorPSet;
  {
    extractorPSet.add<double>("DR_Veto_H",0.1);
    extractorPSet.add<bool>("Vertex_Constraint_Z",false);
    extractorPSet.add<double>("Threshold_H",0.5);
    extractorPSet.add<std::string>("ComponentName","CaloExtractor");
    extractorPSet.add<double>("Threshold_E",0.2);
    extractorPSet.add<double>("DR_Max",1.0);
    extractorPSet.add<double>("DR_Veto_E",0.07);
    extractorPSet.add<double>("Weight_E",1.5);
    extractorPSet.add<bool>("Vertex_Constraint_XY",false);
    extractorPSet.addUntracked<std::string>("DepositLabel","EcalPlusHcal");
    extractorPSet.add<edm::InputTag>("CaloTowerCollectionLabel",edm::InputTag("towerMaker"));
    extractorPSet.add<double>("Weight_H",1.0);
  }
  desc.add<edm::ParameterSetDescription>("ExtractorPSet",extractorPSet);
  edm::ParameterSetDescription isolatorPSet;
  {
    std::vector<double> temp;
    isolatorPSet.add<std::vector<double> >("ConeSizesRel",std::vector<double>(1, 0.3));
    isolatorPSet.add<double>("EffAreaSFEndcap",1.0);
    isolatorPSet.add<bool>("CutAbsoluteIso",true);
    isolatorPSet.add<bool>("AndOrCuts",true);
    isolatorPSet.add<edm::InputTag>("RhoSrc",edm::InputTag("hltKT6CaloJetsForMuons","rho"));
    isolatorPSet.add<std::vector<double> >("ConeSizes",std::vector<double>(1, 0.3));
    isolatorPSet.add<std::string>("ComponentName","CutsIsolatorWithCorrection");
    isolatorPSet.add<bool>("ReturnRelativeSum",false);
    isolatorPSet.add<double>("RhoScaleBarrel",1.0);
    isolatorPSet.add<double>("EffAreaSFBarrel",1.0);
    isolatorPSet.add<bool>("CutRelativeIso",false);
    isolatorPSet.add<std::vector<double> >("EtaBounds",std::vector<double>(1, 2.411));
    isolatorPSet.add<std::vector<double> >("Thresholds",std::vector<double>(1, 9.9999999E7));
    isolatorPSet.add<bool>("ReturnAbsoluteSum",true);
    isolatorPSet.add<std::vector<double> >("EtaBoundsRel",std::vector<double>(1, 2.411));
    isolatorPSet.add<std::vector<double> >("ThresholdsRel",std::vector<double>(1, 9.9999999E7));
    isolatorPSet.add<double>("RhoScaleEndcap",1.0);
    isolatorPSet.add<double>("RhoMax",9.9999999E7);
    isolatorPSet.add<bool>("UseRhoCorrection",true);
  }
  desc.add<edm::ParameterSetDescription>("IsolatorPSet",isolatorPSet);
  desc.add<bool>("WriteIsolatorFloat",false);
  descriptions.add("hltL2MuonIsolations", desc);
}

///beginJob
void L2MuonIsolationProducer::beginJob(){

}

/// build deposits
void L2MuonIsolationProducer::produce(Event& event, const EventSetup& eventSetup){
  std::string metname = "Muon|RecoMuon|L2MuonIsolationProducer";

  LogDebug(metname)<<" L2 Muon Isolation producing...";

  // Take the SA container
  LogDebug(metname)<<" Taking the StandAlone muons: "<<theSACollectionLabel;
  Handle<RecoChargedCandidateCollection> muons;
  event.getByToken(theSACollectionToken,muons);

  // Find deposits and load into event
  LogDebug(metname)<<" Get energy around";
  std::auto_ptr<reco::IsoDepositMap> depMap( new reco::IsoDepositMap());
  std::auto_ptr<edm::ValueMap<bool> > isoMap( new edm::ValueMap<bool> ());
  std::auto_ptr<edm::ValueMap<float> > isoFloatMap( new edm::ValueMap<float> ());

  unsigned int nMuons = muons->size();
  std::vector<IsoDeposit> deps(nMuons);
  std::vector<bool> isos(nMuons, false);
  std::vector<float> isoFloats(nMuons, 0);

  // fill track collection to use for vetos calculation
  TrackCollection muonTracks;
  for (unsigned int i=0; i<nMuons; i++) {
    TrackRef tk = (*muons)[i].track();
    muonTracks.push_back(*tk);
  }

  theExtractor->fillVetos(event,eventSetup,muonTracks);

  for (unsigned int i=0; i<nMuons; i++) {
    TrackRef tk = (*muons)[i].track();

      deps[i] = theExtractor->deposit(event, eventSetup, *tk);

      if (optOutputDecision){
	muonisolation::MuIsoBaseIsolator::DepositContainer isoContainer(1,muonisolation::MuIsoBaseIsolator::DepositAndVetos(&deps[i]));
	muonisolation::MuIsoBaseIsolator::Result isoResult = theDepositIsolator->result( isoContainer, *tk, &event );
	isos[i] = isoResult.valBool;
	isoFloats[i] = isoResult.valFloat;
      }
  }



  //!do the business of filling iso map
  reco::IsoDepositMap::Filler depFiller(*depMap);
  depFiller.insert(muons, deps.begin(), deps.end());
  depFiller.fill();
  event.put(depMap);

  if (optOutputDecision){
    edm::ValueMap<bool> ::Filler isoFiller(*isoMap);
    isoFiller.insert(muons, isos.begin(), isos.end());
    isoFiller.fill();//! annoying -- I will forget it at some point
    event.put(isoMap);

    if (optOutputIsolatorFloat){
      edm::ValueMap<float> ::Filler isoFloatFiller(*isoFloatMap);
      isoFloatFiller.insert(muons, isoFloats.begin(), isoFloats.end());
      isoFloatFiller.fill();//! annoying -- I will forget it at some point
      event.put(isoFloatMap);
    }
  }

  LogDebug(metname) <<" Event loaded"
		    <<"================================";
}

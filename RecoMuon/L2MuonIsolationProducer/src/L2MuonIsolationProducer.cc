/**  \class L2MuonIsolationProducer
 * 
 *   \author  J. Alcaraz
 */

// Framework
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/RecoCandidate/interface/IsoDepositDirection.h"
#include "RecoMuon/L2MuonIsolationProducer/src/L2MuonIsolationProducer.h"

#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
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


  //
  // Extractor
  //
  edm::ParameterSet extractorPSet = par.getParameter<edm::ParameterSet>("ExtractorPSet");
  std::string extractorName = extractorPSet.getParameter<std::string>("ComponentName");
  theExtractor = IsoDepositExtractorFactory::get()->create( extractorName, extractorPSet);  

  
  edm::ParameterSet isolatorPSet = par.getParameter<edm::ParameterSet>("IsolatorPSet");
  bool haveIsolator = !isolatorPSet.empty();
  optOutputDecision = haveIsolator;
  if (optOutputDecision){
    std::string type = isolatorPSet.getParameter<std::string>("ComponentName");
    theDepositIsolator = MuonIsolatorFactory::get()->create(type,isolatorPSet);
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

///beginJob
void L2MuonIsolationProducer::beginJob(){

}

/// build deposits
void L2MuonIsolationProducer::produce(Event& event, const EventSetup& eventSetup){
  std::string metname = "Muon|RecoMuon|L2MuonIsolationProducer";
  
  LogDebug(metname)<<" L2 Muon Isolation producing...";

  // Take the SA container
  LogDebug(metname)<<" Taking the StandAlone muons: "<<theSACollectionLabel;
  Handle<TrackCollection> tracks;
  event.getByLabel(theSACollectionLabel,tracks);

  // Find deposits and load into event
  LogDebug(metname)<<" Get energy around";
  std::auto_ptr<reco::IsoDepositMap> depMap( new reco::IsoDepositMap());
  std::auto_ptr<edm::ValueMap<bool> > isoMap( new edm::ValueMap<bool> ());
  std::auto_ptr<edm::ValueMap<float> > isoFloatMap( new edm::ValueMap<float> ());

  theExtractor->fillVetos(event,eventSetup,*tracks);

  unsigned int nTracks = tracks->size();
  std::vector<IsoDeposit> deps(nTracks);
  std::vector<bool> isos(nTracks, false);
  std::vector<float> isoFloats(nTracks, 0);

  for (unsigned int i=0; i<nTracks; i++) {
      TrackRef tk(tracks,i);

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
  depFiller.insert(tracks, deps.begin(), deps.end());
  depFiller.fill();
  event.put(depMap);

  if (optOutputDecision){
    edm::ValueMap<bool> ::Filler isoFiller(*isoMap);
    isoFiller.insert(tracks, isos.begin(), isos.end());
    isoFiller.fill();//! annoying -- I will forget it at some point
    event.put(isoMap);

    if (optOutputIsolatorFloat){
      edm::ValueMap<float> ::Filler isoFloatFiller(*isoFloatMap);
      isoFloatFiller.insert(tracks, isoFloats.begin(), isoFloats.end());
      isoFloatFiller.fill();//! annoying -- I will forget it at some point
      event.put(isoFloatMap);
    }
  }
  
  LogDebug(metname) <<" Event loaded"
		    <<"================================";
}

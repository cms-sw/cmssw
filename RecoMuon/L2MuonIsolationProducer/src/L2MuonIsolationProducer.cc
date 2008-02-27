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

#include "DataFormats/MuonReco/interface/Direction.h"
#include "RecoMuon/L2MuonIsolationProducer/src/L2MuonIsolationProducer.h"

#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/MuonReco/interface/MuIsoDepositFwd.h"

#include "RecoMuon/MuonIsolation/interface/MuIsoExtractor.h"
#include "RecoMuon/MuonIsolation/interface/MuIsoExtractorFactory.h"

#include <string>

using namespace edm;
using namespace std;
using namespace reco;
using namespace muonisolation;

/// constructor with config
L2MuonIsolationProducer::L2MuonIsolationProducer(const ParameterSet& par) :
  theConfig(par),
  theSACollectionLabel(par.getParameter<edm::InputTag>("StandAloneCollectionLabel")), 
  theCuts(par.getParameter<std::vector<double> > ("EtaBounds"),
          par.getParameter<std::vector<double> > ("ConeSizes"),
          par.getParameter<std::vector<double> > ("Thresholds")),
  optOutputIsoDeposits(par.getParameter<bool>("OutputMuIsoDeposits")),
  theExtractor(0)
{
  LogDebug("Muon|RecoMuon|L2MuonIsolationProducer")<<" L2MuonIsolationProducer constructor called";

  if (optOutputIsoDeposits) produces<reco::IsoDepositMap>();
  produces<reco::MuIsoFlagMap>();
}
  
/// destructor
L2MuonIsolationProducer::~L2MuonIsolationProducer(){
  LogDebug("Muon|RecoMuon|L2MuonIsolationProducer")<<" L2MuonIsolationProducer destructor called";
  if (theExtractor) delete theExtractor;
}

///beginJob
void L2MuonIsolationProducer::beginJob(const edm::EventSetup& iSetup){
  //
  // Extractor
  //
  edm::ParameterSet extractorPSet = theConfig.getParameter<edm::ParameterSet>("ExtractorPSet");
  std::string extractorName = extractorPSet.getParameter<std::string>("ComponentName");
  theExtractor = MuIsoExtractorFactory::get()->create( extractorName, extractorPSet);  
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
  std::auto_ptr<reco::MuIsoFlagMap> isoMap( new reco::MuIsoFlagMap());

  theExtractor->fillVetos(event,eventSetup,*tracks);

  uint nTracks = tracks->size();
  std::vector<MuIsoDeposit> deps(nTracks);
  std::vector<bool> isos(nTracks, false);

  for (unsigned int i=0; i<nTracks; i++) {
      TrackRef tk(tracks,i);

      deps[i] = theExtractor->deposit(event, eventSetup, *tk);

      muonisolation::Cuts::CutSpec cuts_here = theCuts(tk->eta());
      
      double conesize = cuts_here.conesize;
      double dephlt = deps[i].depositWithin(conesize);
      if (dephlt<cuts_here.threshold) {
	isos[i] = true;
      } else {
	isos[i] = false;
      }
  }

  

  if (optOutputIsoDeposits){
    //!do the business of filling iso map
    reco::IsoDepositMap::Filler depFiller(*depMap);
    depFiller.insert(tracks, deps.begin(), deps.end());
    depFiller.fill();
    event.put(depMap);
  }

  reco::MuIsoFlagMap::Filler isoFiller(*isoMap);
  isoFiller.insert(tracks, isos.begin(), isos.end());
  isoFiller.fill();//! annoying -- I will forget it at some point
  event.put(isoMap);
  
  LogDebug(metname) <<" Event loaded"
		    <<"================================";
}

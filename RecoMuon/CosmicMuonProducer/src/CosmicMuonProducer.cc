#include "RecoMuon/CosmicMuonProducer/src/CosmicMuonProducer.h"

/**\class CosmicMuonProducer
 *
 * Description: CosmicMuonProducer for muons from cosmic rays
 *
 * Implementation:
 *
 * $Date: 2006/07/03 01:11:48 $
 * $Revision: 1.6 $
 * Original Author:  Chang Liu
 *        Created:  Tue Jun 13 02:46:17 CEST 2006
**/

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoMuon/CosmicMuonProducer/interface/CosmicMuonTrajectoryBuilder.h"
#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateClosestToPoint.h"
#include "RecoMuon/TrackingTools/interface/MuonTrackFinder.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "RecoMuon/TrackingTools/interface/MuonTrajectoryCleaner.h"
#include "RecoMuon/TrackingTools/interface/MuonTrackLoader.h"

//
// constructors and destructor
//
CosmicMuonProducer::CosmicMuonProducer(const edm::ParameterSet& iConfig)
{

  edm::ParameterSet tbpar = iConfig.getParameter<edm::ParameterSet>("TrajectoryBuilderParameters");
  theSeedCollectionLabel = iConfig.getParameter<std::string>("MuonSeedCollectionLabel");
  theTrackFinder = new MuonTrackFinder(new CosmicMuonTrajectoryBuilder(tbpar));

  produces<reco::TrackCollection>();
  produces<TrackingRecHitCollection>();
  produces<reco::TrackExtraCollection>();

  edm::LogInfo("CosmicMuonProducer")<<"CosmicMuonProducer begin";

}


CosmicMuonProducer::~CosmicMuonProducer()
{
  if (theTrackFinder) delete theTrackFinder;
}


// ------------ method called to produce the data  ------------
void
CosmicMuonProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::LogInfo("CosmicMuonProducer") << "Analyzing event number: " << iEvent.id();


  edm::Handle<TrajectorySeedCollection> seeds; 
  iEvent.getByLabel(theSeedCollectionLabel,seeds);

  theTrackFinder->reconstruct(seeds,iEvent,iSetup);

}

//define this as a plug-in
DEFINE_FWK_MODULE(CosmicMuonProducer)

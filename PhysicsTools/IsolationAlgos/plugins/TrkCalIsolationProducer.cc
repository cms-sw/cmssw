/* \class TrkCalIsolationProducer
 *
 * computes and stores calorimeter isolation using CalIsolationAlgo for Tracks
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/IsolationAlgos/interface/IsolationProducer.h"
#include "PhysicsTools/IsolationAlgos/interface/TrkCalIsolationAlgo.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"

typedef IsolationProducer<reco::TrackCollection, CaloTowerCollection,
			  TrkCalIsolationAlgo<reco::Track, CaloTowerCollection> > 
                          TrkCalIsolationProducer;

DEFINE_FWK_MODULE( TrkCalIsolationProducer );

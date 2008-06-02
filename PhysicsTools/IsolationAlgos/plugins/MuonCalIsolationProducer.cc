/* \class MuonCalIsolationProducer
 *
 * computes and stores calorimeter isolation using CalIsolationAlgo for Muons
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/IsolationAlgos/interface/IsolationProducer.h"
#include "PhysicsTools/IsolationAlgos/interface/CalIsolationExtrapolate.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"

typedef IsolationProducer<reco::MuonCollection, CaloTowerCollection,
			  CalIsolationAlgo<reco::Muon, CaloTowerCollection>
                         > MuonCalIsolationProducer;

DEFINE_FWK_MODULE( MuonCalIsolationProducer );

/* \class CandPtIsolationProducerNew
 *
 * computes and stores isolation using PtAlgo for Candidates
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "PhysicsTools/IsolationAlgos/interface/IsolationProducerNew.h"
#include "PhysicsTools/IsolationAlgos/interface/PtIsolationAlgo.h"

typedef reco::modulesNew::IsolationProducer<reco::CandidateView, reco::TrackCollection,
					    PtIsolationAlgo<reco::Candidate,reco::TrackCollection>
                                            > CandTrackPtIsolationProducer;

DEFINE_FWK_MODULE(CandTrackPtIsolationProducer);

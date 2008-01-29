/* \class CandPtIsolationProducerNew
 *
 * computes and stores isolation using PtAlgo for Candidates
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "PhysicsTools/IsolationAlgos/interface/IsolationProducerNew.h"
#include "PhysicsTools/IsolationAlgos/interface/PtIsolationAlgo.h"

typedef reco::modulesNew::IsolationProducer<reco::CandidateView, reco::CandidateView,
					    PtIsolationAlgo<reco::Candidate,reco::CandidateView>
                                            > CandPtIsolationProducerNew;

DEFINE_FWK_MODULE( CandPtIsolationProducerNew );

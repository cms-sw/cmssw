/* \class CandViewPtIsolationProducer
 *
 * computes and stores isolation using PtAlgo for Candidates
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "PhysicsTools/IsolationAlgos/interface/IsolationProducer.h"
#include "PhysicsTools/IsolationAlgos/interface/PtIsolationAlgo.h"

typedef IsolationProducer<reco::CandidateView, reco::CandidateView,
			  PtIsolationAlgo<reco::Candidate,reco::CandidateView>,
			  edm::AssociationVector<reco::CandidateBaseRefProd, std::vector<float> > 
                         > CandViewPtIsolationProducer;

DEFINE_FWK_MODULE( CandViewPtIsolationProducer );

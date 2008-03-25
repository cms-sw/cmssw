/** \class EcalCandidateProducer
 *
 * Framework module that produces a collection
 * of candidates with a SuperCluster compoment
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.1 $
 *
 * $Id: EcalCandidateProducer.cc,v 1.1 2007/02/01 13:11:18 llista Exp $
 *
 */
#include "PhysicsTools/RecoAlgos/src/SuperClusterToCandidate.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef CandidateProducer<
          reco::SuperClusterCollection, 
          reco::CandidateCollection
        > EcalCandidateProducer;

DEFINE_FWK_MODULE( EcalCandidateProducer );


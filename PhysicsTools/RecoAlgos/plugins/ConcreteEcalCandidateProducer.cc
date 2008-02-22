/** \class ConcreteEcalCandidateProducer
 *
 * Framework module that produces a collection
 * of candidates with a Ecal compoment
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.1 $
 *
 * $Id: ConcreteChargedCandidateProducer.cc,v 1.1 2007/10/31 15:08:06 llista Exp $
 *
 */

#include "PhysicsTools/RecoAlgos/src/SuperClusterToCandidate.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef CandidateProducer<
          reco::SuperClusterCollection, 
          reco::RecoEcalCandidateCollection
        > ConcreteEcalCandidateProducer;

DEFINE_FWK_MODULE( ConcreteEcalCandidateProducer );

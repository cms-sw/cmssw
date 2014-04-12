/** \class ConcreteEcalCandidateProducer
 *
 * Framework module that produces a collection
 * of candidates with a Ecal compoment
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.1 $
 *
 * $Id: ConcreteEcalCandidateProducer.cc,v 1.1 2009/03/04 13:11:29 llista Exp $
 *
 */

#include "CommonTools/RecoAlgos/src/SuperClusterToCandidate.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef CandidateProducer<
          reco::SuperClusterCollection, 
          reco::RecoEcalCandidateCollection
        > ConcreteEcalCandidateProducer;

DEFINE_FWK_MODULE( ConcreteEcalCandidateProducer );

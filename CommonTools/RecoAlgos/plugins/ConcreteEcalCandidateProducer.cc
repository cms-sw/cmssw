/** \class ConcreteEcalCandidateProducer
 *
 * Framework module that produces a collection
 * of candidates with a Ecal compoment
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.2 $
 *
 * $Id: ConcreteEcalCandidateProducer.cc,v 1.2 2013/02/28 00:17:18 wmtan Exp $
 *
 */

#include "CommonTools/RecoAlgos/src/SuperClusterToCandidate.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef CandidateProducer<
          reco::SuperClusterCollection, 
          reco::RecoEcalCandidateCollection
        > ConcreteEcalCandidateProducer;

DEFINE_FWK_MODULE( ConcreteEcalCandidateProducer );

/* \class 
 * 
 * Producer of merged Candidate collection 
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/CandAlgos/interface/ShallowCloneProducer.h"
#include "DataFormats/Candidate/interface/Candidate.h"

typedef ShallowCloneProducer<
          reco::CandidateCollection
        > CandShallowCloneProducer;

DEFINE_FWK_MODULE( CandShallowCloneProducer );

/* \class CandCandAssociationMapOneToOne2Association
 * 
 * Configurable Candidate-to-double AssociationVector 2ValueMap
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/AssociationMapOneToOne2Association.h"
#include "DataFormats/Candidate/interface/Candidate.h"

typedef AssociationMapOneToOne2Association<
          reco::CandidateCollection,
          reco::CandidateCollection
        > CandCandAssociationMapOneToOne2Association;

DEFINE_FWK_MODULE(CandCandAssociationMapOneToOne2Association);

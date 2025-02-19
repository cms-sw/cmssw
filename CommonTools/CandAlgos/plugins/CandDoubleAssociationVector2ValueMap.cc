/* \class CandDoubleAssociationVector2ValueMap
 * 
 * Configurable Candidate-to-double AssociationVector 2ValueMap
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/AssociationVector2ValueMap.h"
#include "DataFormats/Candidate/interface/Candidate.h"

typedef AssociationVector2ValueMap<
          reco::CandidateRefProd,
          std::vector<double>
        > CandDoubleAssociationVector2ValueMap;

DEFINE_FWK_MODULE(CandDoubleAssociationVector2ValueMap);

/* \class CandDoubleAssociationVectorSelector
 * 
 * Configurable Candidate-to-double AssociationVector Selector
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/AssociationVectorSelector.h"
#include "CommonTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/MaxSelector.h"
#include "DataFormats/Candidate/interface/Candidate.h"

typedef AssociationVectorSelector<
          reco::CandidateRefProd,
          std::vector<double>,
          StringCutObjectSelector<reco::Candidate>,
          MaxSelector<double>
        > CandDoubleAssociationVectorSelector;

DEFINE_FWK_MODULE(CandDoubleAssociationVectorSelector);

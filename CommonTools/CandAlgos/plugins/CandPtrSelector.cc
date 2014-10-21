/* \class CandPtrSelector
 * 
 * Candidate Selector based on a configurable cut.
 * Reads a edm::View<Candidate> as input
 * and saves a vector of references (edm::PtrVector)
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "DataFormats/Candidate/interface/Candidate.h"

typedef SingleObjectSelector<
          edm::View<reco::Candidate>,
          StringCutObjectSelector<reco::Candidate, true>,
          edm::PtrVector<reco::Candidate>
       > CandPtrSelector;

DEFINE_FWK_MODULE(CandPtrSelector);

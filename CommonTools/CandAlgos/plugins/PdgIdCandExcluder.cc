/* \class PdgIdCandExcluder
 * 
 * Candidate Excluder based on a pdgId set
 * Usage:
 * 
 * module allExceptLeptons = PdgIdCandExcluder {
 *   InputTag src = myCollection
 *   vint32 pdgId = { 11, 13 }
 * };
 *
 * \author: Loic Quertenmont, UCL
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/PdgIdExcluder.h"
#include "DataFormats/Candidate/interface/Candidate.h"

typedef SingleObjectSelector<
  reco::CandidateCollection,
          PdgIdExcluder
  > PdgIdCandExcluder;

DEFINE_FWK_MODULE( PdgIdCandExcluder );

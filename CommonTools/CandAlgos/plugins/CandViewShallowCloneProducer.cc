/* \class CandViewShallowCloneProducer
 * 
 * Producer of ShallowClones from any candidate collection   
 * selection via the string parser
 *
 * \author: Benedikt Hegner, CERN
 *
 */

#include "CommonTools/CandAlgos/interface/SingleObjectShallowCloneSelector.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "CommonTools/UtilAlgos/interface/StringCutObjectSelector.h"

typedef SingleObjectShallowCloneSelector<edm::View<reco::Candidate>, StringCutObjectSelector<reco::Candidate> > CandViewShallowCloneProducer;

DEFINE_FWK_MODULE(CandViewShallowCloneProducer);



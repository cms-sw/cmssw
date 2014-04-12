/* \class CandHistoAnalyzer
 * 
 * Configurable Candidate Histogram creator
 *
 * \author: Benedikt Hegner, DESY
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/HistoAnalyzer.h"
#include "DataFormats/Candidate/interface/Candidate.h"

typedef HistoAnalyzer<reco::CandidateCollection> CandHistoAnalyzer;

DEFINE_FWK_MODULE( CandHistoAnalyzer );


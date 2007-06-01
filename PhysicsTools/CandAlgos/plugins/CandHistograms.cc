/* \class reco::modules::CandHistograms
 * 
 * Configurable Candidate Histogram creator
 *
 * \author: Benedikt Hegner, DESY
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/CandAlgos/interface/HistoAnalyzer.h"
#include "DataFormats/Candidate/interface/Candidate.h"


typedef HistoAnalyzer<reco::CandidateCollection> CandHistograms;

DEFINE_FWK_MODULE( CandHistograms );


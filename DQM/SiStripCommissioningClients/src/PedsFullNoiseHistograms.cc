#include "DQM/SiStripCommissioningClients/interface/PedsFullNoiseHistograms.h"
#include "CondFormats/SiStripObjects/interface/PedsFullNoiseAnalysis.h"
#include "DQM/SiStripCommissioningAnalysis/interface/PedsFullNoiseAlgorithm.h"
#include "DQM/SiStripCommissioningSummary/interface/PedsFullNoiseSummaryFactory.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DQM/SiStripCommon/interface/ExtractTObject.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include "TProfile.h"

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
/** */
PedsFullNoiseHistograms::PedsFullNoiseHistograms( const edm::ParameterSet& pset,
                                                  DQMStore* bei ) 
  : CommissioningHistograms( pset.getParameter<edm::ParameterSet>("PedsFullNoiseParameters"),
                             bei,
                             sistrip::PEDS_FULL_NOISE )
{

  factory_ = unique_ptr<PedsFullNoiseSummaryFactory>( new PedsFullNoiseSummaryFactory );
  LogTrace(mlDqmClient_) 
    << "[PedsFullNoiseHistograms::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
PedsFullNoiseHistograms::~PedsFullNoiseHistograms() {
  LogTrace(mlDqmClient_) 
    << "[PedsFullNoiseHistograms::" << __func__ << "]"
    << " Destructing object...";
}

// -----------------------------------------------------------------------------	 
/** */	 
void PedsFullNoiseHistograms::histoAnalysis( bool debug ) {
  
  LogTrace(mlDqmClient_)
    << "[PedsFullNoiseHistograms::" << __func__ << "]";

  // Some initialisation
  uint16_t valid = 0;
  HistosMap::const_iterator iter;
  Analyses::iterator ianal;
  std::map<std::string,uint16_t> errors;

  // Clear map holding analysis objects
  for ( ianal = data().begin(); ianal != data().end(); ianal++ ) { 
    if ( ianal->second ) { delete ianal->second; }
  } 
  data().clear();

  // Iterate through map containing histograms
  long int ichannel = 0;
  long int nchannel = histos().size();
  for ( iter = histos().begin(); 
	iter != histos().end(); iter++ ) {

    // Check vector of histos is not empty
    if ( iter->second.empty() ) {
      edm::LogWarning(mlDqmClient_) 
	<< "[PedsFullNoiseHistograms::" << __func__ << "]"
	<< " Zero histograms found!";
      continue;
    }
    
    // Retrieve pointers to peds and noise histos
    std::vector<TH1*> hists;
    Histos::const_iterator ihis = iter->second.begin(); 

    for ( ; ihis != iter->second.end(); ihis++ ) {
      // pedestal and noise 1D profiles
      TProfile* prof = ExtractTObject<TProfile>().extract( (*ihis)->me_ );
      if ( prof ) { hists.push_back(prof);
      }
      // 2D noise histograms
      TH2S * his2D = ExtractTObject<TH2S>().extract( (*ihis)->me_ );
      if ( his2D ) { 
	hists.push_back(his2D); }
    }

    if(ichannel % 100 == 0)
      edm::LogVerbatim(mlDqmClient_)
	<< "[PedsFullNoiseHistograms::" << __func__ << "]"
	<< " Analyzing channel " << ichannel << " out of "<<nchannel;
    ichannel++;

    // Perform histo analysis
    PedsFullNoiseAnalysis * anal = new PedsFullNoiseAnalysis( iter->first );
    PedsFullNoiseAlgorithm algo( this->pset(), anal );    
    algo.analysis( hists );

    data()[iter->first] = anal; 
    if (anal->isValid() ) { valid++; }
    if (!anal->getErrorCodes().empty() ) { 
      errors[anal->getErrorCodes()[0]]++;
    } 
    
  }

  if ( !histos().empty() ) {
    edm::LogVerbatim(mlDqmClient_) 
      << "[PedsFullNoiseHistograms::" << __func__ << "]"
      << " Analyzed histograms for " << histos().size() 
      << " FED channels, of which " << valid 
      << " (" << 100 * valid / histos().size()
      << "%) are valid.";
    if ( !errors.empty() ) {
      uint16_t count = 0;
      std::stringstream ss;
      ss << std::endl;
      std::map<std::string,uint16_t>::const_iterator ii;
      for ( ii = errors.begin(); ii != errors.end(); ++ii ) { 
	ss << " " << ii->first << ": " << ii->second << std::endl;
	count += ii->second;
      }
      edm::LogWarning(mlDqmClient_) 
	<< "[PedsFullNoiseHistograms::" << __func__ << "]"
	<< " Found " << count << " errors ("
	<< 100 * count / histos().size() << "%): " 
	<< ss.str();
    }
  } else {
    edm::LogWarning(mlDqmClient_) 
      << "[PedsFullNoiseHistograms::" << __func__ << "]"
      << " No histograms to analyze!";
  }  
   
}
  
// -----------------------------------------------------------------------------	 
/** */	 
void PedsFullNoiseHistograms::printAnalyses() {
  Analyses::iterator ianal = data().begin();
  Analyses::iterator janal = data().end();
  for ( ; ianal != janal; ++ianal ) { 
    if ( ianal->second ) { 
      std::stringstream ss;
      ianal->second->print( ss, 1 ); 
      ianal->second->print( ss, 2 ); 
      if ( ianal->second->isValid() ) { LogTrace(mlDqmClient_) << ss.str(); 
      } else { edm::LogWarning(mlDqmClient_) << ss.str(); }
    }
  }
}

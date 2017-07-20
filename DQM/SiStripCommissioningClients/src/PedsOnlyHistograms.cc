#include "DQM/SiStripCommissioningClients/interface/PedsOnlyHistograms.h"
#include "CondFormats/SiStripObjects/interface/PedsOnlyAnalysis.h"
#include "DQM/SiStripCommissioningAnalysis/interface/PedsOnlyAlgorithm.h"
#include "DQM/SiStripCommissioningSummary/interface/PedsOnlySummaryFactory.h"
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
PedsOnlyHistograms::PedsOnlyHistograms( const edm::ParameterSet& pset,
                                        DQMStore* bei ) 
  : CommissioningHistograms( pset.getParameter<edm::ParameterSet>("PedsOnlyParameters"),
                             bei,
                             sistrip::PEDS_ONLY )
{
  factory_ = unique_ptr<PedsOnlySummaryFactory>( new PedsOnlySummaryFactory );
  LogTrace(mlDqmClient_) 
    << "[PedsOnlyHistograms::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
PedsOnlyHistograms::~PedsOnlyHistograms() {
  LogTrace(mlDqmClient_) 
    << "[PedsOnlyHistograms::" << __func__ << "]"
    << " Destructing object...";
}

// -----------------------------------------------------------------------------	 
/** */	 
void PedsOnlyHistograms::histoAnalysis( bool debug ) {
  LogTrace(mlDqmClient_)
    << "[PedsOnlyHistograms::" << __func__ << "]";

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
  for ( iter = histos().begin(); 
	iter != histos().end(); iter++ ) {
    
    // Check vector of histos is not empty
    if ( iter->second.empty() ) {
      edm::LogWarning(mlDqmClient_) 
	<< "[PedsOnlyHistograms::" << __func__ << "]"
	<< " Zero histograms found!";
      continue;
    }
    
    // Retrieve pointers to profile histos
    std::vector<TH1*> profs;
    Histos::const_iterator ihis = iter->second.begin(); 
    for ( ; ihis != iter->second.end(); ihis++ ) {
      TProfile* prof = ExtractTObject<TProfile>().extract( (*ihis)->me_ );
      if ( prof ) { profs.push_back(prof); }
      //@@ Common mode histos?...
      //TH1F* his = ExtractTObject<TH1F>().extract( (*ihis)->me_ );
      //if ( his ) { profs.push_back(his); }
    }
    
    // Perform histo analysis
    PedsOnlyAnalysis* anal = new PedsOnlyAnalysis( iter->first );
    PedsOnlyAlgorithm algo( this->pset(), anal );
    algo.analysis( profs );
    data()[iter->first] = anal; 
    if ( anal->isValid() ) { valid++; }
    if ( !anal->getErrorCodes().empty() ) { 
      errors[anal->getErrorCodes()[0]]++;
    }
    
  }
  
  if ( !histos().empty() ) {
    edm::LogVerbatim(mlDqmClient_) 
      << "[PedsOnlyHistograms::" << __func__ << "]"
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
	<< "[PedsOnlyHistograms::" << __func__ << "]"
	<< " Found " << count << " errors ("
	<< 100 * count / histos().size() << "%): " 
	<< ss.str();
    }
  } else {
    edm::LogWarning(mlDqmClient_) 
      << "[PedsOnlyHistograms::" << __func__ << "]"
      << " No histograms to analyze!";
  }
  
}

// -----------------------------------------------------------------------------	 
/** */	 
void PedsOnlyHistograms::printAnalyses() {
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

#include "DQM/SiStripCommissioningClients/interface/PedestalsHistograms.h"
#include "CondFormats/SiStripObjects/interface/PedestalsAnalysis.h"
#include "DQM/SiStripCommissioningAnalysis/interface/PedestalsAlgorithm.h"
#include "DQM/SiStripCommissioningSummary/interface/PedestalsSummaryFactory.h"
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
PedestalsHistograms::PedestalsHistograms( const edm::ParameterSet& pset,
                                          DQMStore* bei ) 
  : CommissioningHistograms( pset.getParameter<edm::ParameterSet>("PedestalsParameters"),
                             bei,
                             sistrip::PEDESTALS )
{
  factory_ = unique_ptr<PedestalsSummaryFactory>( new PedestalsSummaryFactory );
  LogTrace(mlDqmClient_) 
    << "[PedestalsHistograms::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
PedestalsHistograms::~PedestalsHistograms() {
  LogTrace(mlDqmClient_) 
    << "[PedestalsHistograms::" << __func__ << "]"
    << " Destructing object...";
}

// -----------------------------------------------------------------------------	 
/** */	 
void PedestalsHistograms::histoAnalysis( bool debug ) {
  LogTrace(mlDqmClient_)
    << "[PedestalsHistograms::" << __func__ << "]";

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
	<< "[PedestalsHistograms::" << __func__ << "]"
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
    PedestalsAnalysis* anal = new PedestalsAnalysis( iter->first );
    PedestalsAlgorithm algo( this->pset(), anal );
    algo.analysis( profs );
    data()[iter->first] = anal; 
    if ( anal->isValid() ) { valid++; }
    if ( !anal->getErrorCodes().empty() ) { 
      errors[anal->getErrorCodes()[0]]++;
    }
    
  }
  
  if ( !histos().empty() ) {
    edm::LogVerbatim(mlDqmClient_) 
      << "[PedestalsHistograms::" << __func__ << "]"
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
	<< "[PedestalsHistograms::" << __func__ << "]"
	<< " Found " << count << " errors ("
	<< 100 * count / histos().size() << "%): " 
	<< ss.str();
    }
  } else {
    edm::LogWarning(mlDqmClient_) 
      << "[PedestalsHistograms::" << __func__ << "]"
      << " No histograms to analyze!";
  }
  
}

// -----------------------------------------------------------------------------	 
/** */	 
void PedestalsHistograms::printAnalyses() {
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

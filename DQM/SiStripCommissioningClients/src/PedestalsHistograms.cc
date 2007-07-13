#include "DQM/SiStripCommissioningClients/interface/PedestalsHistograms.h"
#include "DQM/SiStripCommissioningSummary/interface/SummaryGenerator.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <sstream>
#include <iomanip>

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
/** */
PedestalsHistograms::PedestalsHistograms( MonitorUserInterface* mui ) 
  : CommissioningHistograms( mui, sistrip::PEDESTALS ),
    factory_( new Factory )
{
  LogTrace(mlDqmClient_) 
    << "[PedestalsHistograms::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
PedestalsHistograms::PedestalsHistograms( DaqMonitorBEInterface* bei ) 
  : CommissioningHistograms( bei, sistrip::PEDESTALS ),
    factory_( new Factory )
{
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

  // Some initialisation
  uint16_t valid = 0;
  HistosMap::const_iterator iter = 0;
  Analyses::iterator ianal = 0;
  std::map<std::string,uint16_t> errors;

  // Clear map holding analysis objects
  for ( ianal = data_.begin(); ianal != data_.end(); ianal++ ) { 
    if ( ianal->second ) { delete ianal->second; }
  } 
  data_.clear();
  
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
    }
    
    // Perform histo analysis
    PedestalsAnalysis* anal = new PedestalsAnalysis( iter->first );
    anal->analysis( profs );
    data_[iter->first] = anal; 
    data_[iter->first] = anal; 
    if ( anal->isValid() ) { valid++; }
    if ( debug ) {
      std::stringstream ss;
      anal->print( ss, 1 ); 
      anal->print( ss, 2 ); 
      if ( anal->isValid() ) { LogTrace(mlDqmClient_) << ss.str(); 
      } else { edm::LogWarning(mlDqmClient_) << ss.str(); }
      if ( !anal->getErrorCodes().empty() ) { 
	errors[anal->getErrorCodes()[0]]++;
      }
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
      edm::LogVerbatim(mlDqmClient_) 
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
void PedestalsHistograms::createSummaryHisto( const sistrip::Monitorable& mon, 
					      const sistrip::Presentation& pres, 
					      const std::string& dir,
					      const sistrip::Granularity& gran ) {

  // Analyze histograms if not done already
  if ( data_.empty() ) { histoAnalysis( false ); }
  
  // Extract data to be histogrammed
  sistrip::View view = SiStripEnumsAndStrings::view(dir);
  uint32_t xbins = factory_->init( mon, pres, view, dir, gran, data_ );
  
  // Use base method to create summary histogram
  TH1* summary = 0;
  if ( pres != sistrip::HISTO_1D ) { summary = histogram( mon, pres, view, dir, xbins ); }
  else { summary = histogram( mon, pres, view, dir, sistrip::FED_ADC_RANGE, 0., sistrip::FED_ADC_RANGE*1. ); }
  
  // Fill histogram with data
  factory_->fill( *summary );
  
}

#include "DQM/SiStripCommissioningClients/interface/CalibrationHistograms.h"
#include "DQM/SiStripCommissioningSummary/interface/SummaryGenerator.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "DQM/SiStripCommon/interface/ExtractTObject.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include "TProfile.h"
 
using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
/** */
CalibrationHistograms::CalibrationHistograms( DQMStore* bei,const sistrip::RunType& task ) 
  : CommissioningHistograms( bei, task ),
    factory_( new Factory ),
    calchan_(0)
{
  cout << endl // LogTrace(mlDqmClient_) 
       << "[CalibrationHistograms::" << __func__ << "]"
       << " Constructing object...";
  std::string pwd = bei->pwd();
  std::string calchanPath = pwd.substr(0,pwd.find(sistrip::root_ + "/")+sistrip::root_.size()+1);
  calchanPath += "/calchan";
  MonitorElement* calchanElement = bei->get(calchanPath);
  if(calchanElement) {
    calchan_ = calchanElement->getIntValue();
    edm::LogVerbatim(mlDqmClient_)
      << "[CalibrationHistograms::" << __func__ << "]"
      << "CalChan value is " << calchan_;
  } else {
    edm::LogWarning(mlDqmClient_)
      << "[CalibrationHistograms::" << __func__ << "]"
      << "CalChan value not found at " << calchanPath
      << ". Using " << calchan_;
  }
}

CalibrationHistograms::CalibrationHistograms( DQMOldReceiver* mui,const sistrip::RunType& task ) 
  : CommissioningHistograms( mui, task ),
    factory_( new Factory ),
    calchan_(0)
{
  cout << endl // LogTrace(mlDqmClient_) 
       << "[CalibrationHistograms::" << __func__ << "]"
       << " Constructing object...";
  std::string pwd = bei()->pwd();
  std::string calchanPath = pwd.substr(0,pwd.find(sistrip::root_ + "/")+sistrip::root_.size()+1);
  calchanPath += "/calchan";
  MonitorElement* calchanElement = bei()->get(calchanPath);
  if(calchanElement) {
    calchan_ = calchanElement->getIntValue() ;
    edm::LogVerbatim(mlDqmClient_)
      << "[CalibrationHistograms::" << __func__ << "]"
      << "CalChan value is " << calchan_;
  } else {
    edm::LogWarning(mlDqmClient_)
      << "[CalibrationHistograms::" << __func__ << "]"
      << "CalChan value not found at " << calchanPath
      << ". Using " << calchan_;
  }
}

// -----------------------------------------------------------------------------
/** */
CalibrationHistograms::~CalibrationHistograms() {
  cout << endl // LogTrace(mlDqmClient_) 
       << "[CalibrationHistograms::" << __func__ << "]"
       << " Deleting object...";
}

// -----------------------------------------------------------------------------	 
/** */	 
void CalibrationHistograms::histoAnalysis( bool debug ) {
  Analyses::iterator ianal;
  // Clear map holding analysis objects
  for ( ianal = data_.begin(); ianal != data_.end(); ianal++ ) {
    if ( ianal->second ) { delete ianal->second; }
  }
  data_.clear();
  
  // Iterate through map containing vectors of profile histograms
  HistosMap::const_iterator iter = histos().begin();
  for ( ; iter != histos().end(); iter++ ) {
    // Check vector of histos is not empty (should be 1 histo)
    if ( iter->second.empty() ) {
      edm::LogWarning(mlDqmClient_)
 	   << "[CalibrationHistograms::" << __func__ << "]"
 	   << " Zero collation histograms found!";
      continue;
    }
    
    // Retrieve pointers to 1D histos for this FED channel 
    vector<TH1*> profs;
    Histos::const_iterator ihis = iter->second.begin();
    for ( ; ihis != iter->second.end(); ihis++ ) {
      TH1F* prof = ExtractTObject<TH1F>().extract( (*ihis)->me_ );
      if ( prof ) { profs.push_back(prof); }
    } 
    // Perform histo analysis 
    CalibrationAnalysis* anal = new CalibrationAnalysis( iter->first, (task()==sistrip::CALIBRATION_DECO), calchan_ );
    anal->analysis( profs );
    data_[iter->first] = anal; 
    
 }
 
}

// -----------------------------------------------------------------------------
/** */
void CalibrationHistograms::createSummaryHisto( const sistrip::Monitorable& mon, 
					      const sistrip::Presentation& pres, 
					      const string& directory,
					      const sistrip::Granularity& gran ) {


  cout << endl // LogTrace(mlDqmClient_)
       << "[CalibrationHistograms::" << __func__ << "]";
  
  // Check view 
  sistrip::View view = SiStripEnumsAndStrings::view(directory);
  if ( view == sistrip::UNKNOWN_VIEW ) { return; }

  // Analyze histograms if not done already
  if ( data_.empty() ) { histoAnalysis( false ); }

  // Extract data to be histogrammed
  uint32_t xbins = factory_->init( mon, pres, view, directory, gran, data_ );

  // Create summary histogram (if it doesn't already exist)
  TH1* summary = 0;
  if ( pres != sistrip::HISTO_1D ) { summary = histogram( mon, pres, view, directory, xbins ); }
  else { summary = histogram( mon, pres, view, directory, sistrip::FED_ADC_RANGE, 0., sistrip::FED_ADC_RANGE*1. ); }

  // Fill histogram with data
  factory_->fill( *summary );
  
}

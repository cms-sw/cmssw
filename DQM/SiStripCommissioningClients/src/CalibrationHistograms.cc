#include "DQM/SiStripCommissioningClients/interface/CalibrationHistograms.h"
#include "CondFormats/SiStripObjects/interface/CalibrationAnalysis.h"
#include "DQM/SiStripCommissioningAnalysis/interface/CalibrationAlgorithm.h"
#include "DQM/SiStripCommissioningSummary/interface/CalibrationSummaryFactory.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DQM/SiStripCommon/interface/ExtractTObject.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include "TH1F.h"

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
/** */
CalibrationHistograms::CalibrationHistograms( DQMStore* bei,const sistrip::RunType& task ) 
  : CommissioningHistograms( bei, task ),
    calchan_(0),
    isha_(-1),
    vfs_(-1)
{
  LogTrace(mlDqmClient_) 
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
  std::string ishaPath = pwd.substr(0,pwd.find(sistrip::root_ + "/")+sistrip::root_.size()+1);
  ishaPath += "/isha";
  MonitorElement* ishaElement = bei->get(ishaPath);
  if(ishaElement) isha_ = ishaElement->getIntValue() ;
  std::string vfsPath = pwd.substr(0,pwd.find(sistrip::root_ + "/")+sistrip::root_.size()+1);
  vfsPath += "/vfs";
  MonitorElement* vfsElement = bei->get(vfsPath);
  if(vfsElement) vfs_ = vfsElement->getIntValue() ;
}

CalibrationHistograms::CalibrationHistograms( DQMOldReceiver* mui,const sistrip::RunType& task ) 
  : CommissioningHistograms( mui, task ),
    calchan_(0),
    isha_(-1),
    vfs_(-1)
{
  LogTrace(mlDqmClient_) 
       << "[CalibrationHistograms::" << __func__ << "]"
       << " Constructing object...";
  factory_ = auto_ptr<CalibrationSummaryFactory>( new CalibrationSummaryFactory );
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
  std::string ishaPath = pwd.substr(0,pwd.find(sistrip::root_ + "/")+sistrip::root_.size()+1);
  ishaPath += "/isha";
  MonitorElement*  ishaElement = bei()->get(ishaPath);
  if(ishaElement) isha_ = ishaElement->getIntValue() ;
  std::string vfsPath = pwd.substr(0,pwd.find(sistrip::root_ + "/")+sistrip::root_.size()+1);
  vfsPath += "/vfs";
  MonitorElement*  vfsElement = bei()->get(vfsPath);
  if(vfsElement) vfs_ = vfsElement->getIntValue() ;
}

// -----------------------------------------------------------------------------
/** */
CalibrationHistograms::~CalibrationHistograms() {
  LogTrace(mlDqmClient_) 
       << "[CalibrationHistograms::" << __func__ << "]"
       << " Deleting object...";
}

// -----------------------------------------------------------------------------	 
/** */	 
void CalibrationHistograms::histoAnalysis( bool debug ) {

  // Clear map holding analysis objects
  Analyses::iterator ianal;
  for ( ianal = data().begin(); ianal != data().end(); ianal++ ) {
    if ( ianal->second ) { delete ianal->second; }
  }
  data().clear();
  
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
    CalibrationAlgorithm algo( anal );
    algo.analysis( profs );
    data()[iter->first] = anal; 
    
 }
 
}


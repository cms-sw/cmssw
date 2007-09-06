#include "DQM/SiStripCommissioningSources/interface/CalibrationTask.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// -----------------------------------------------------------------------------
//
CalibrationTask::CalibrationTask( DaqMonitorBEInterface* dqm,
			      const FedChannelConnection& conn,
			      const sistrip::RunType& rtype,
			      const char* filename ) :
  CommissioningTask( dqm, conn, "CalibrationTask" ),
  runType_(rtype),
  nBins_(65),lastCalChan_(0),
  filename_(filename)
{
  LogDebug("Commissioning") << "[CalibrationTask::CalibrationTask] Constructing object...";
}

// -----------------------------------------------------------------------------
//
CalibrationTask::~CalibrationTask() {
  LogDebug("Commissioning") << "[CalibrationTask::CalibrationTask] Destructing object...";
}

// -----------------------------------------------------------------------------
//
void CalibrationTask::book() {
  LogDebug("Commissioning") << "[CalibrationTask::book]";

  // book 32 histograms, one for each strip in a calibration group on 2 APVs
  calib_.resize(32);
  for(int apv=0;apv<2;++apv) {
    for(int i=0;i<16;++i) {
      std::string title = SiStripHistoTitle( sistrip::EXPERT_HISTO, 
                                             runType_, 
                                             sistrip::DET_KEY, 
                                             connection().detId(),
                                             sistrip::APV, 
                                             connection().i2cAddr(apv) ).title(); 
      calib_[apv*16+i].histo_ = dqm()->book1D( title, title, nBins_, 0, 203.125);
      calib_[apv*16+i].isProfile_=false;
      calib_[apv*16+i].vNumOfEntries_.resize(nBins_,0);
//  calib[apv*16+i]1_.vSumOfContents_.resize(nBins_,0);
//  calib[apv*16+i]1_.vSumOfSquares_.resize(nBins_,0);
    }
  }
  LogDebug("Commissioning") << "[CalibrationTask::book] done";
  
}

// -----------------------------------------------------------------------------
//
void CalibrationTask::fill( const SiStripEventSummary& summary,
			    const edm::DetSet<SiStripRawDigi>& digis ) {
  LogDebug("Commissioning") << "[CalibrationTask::fill]";
  // Check if CalChan changed. In that case, save, reset histo, change title, and continue
  checkAndSave(summary.calChan());
  // retrieve the delay from the EventSummary
  int bin = (100-summary.latency())*8+(7-summary.calSel());
  if(runType_==sistrip::CALIBRATION_SCAN) bin +=16; // difference between PEAK and DECO
  int isub,ical = summary.calChan();
  isub = ical<4 ? ical+4 : ical-4;
  for (int apv=0; apv<2; ++apv) {
    for (int k=0;k<16;++k)  {
      //TODO: we must substract the pedestal (simplest would be to read it when constructing the task)
      updateHistoSet( calib_[apv*16+ical+k*8],bin,digis.data[apv*128+k].adc());
    }
  }
}

// -----------------------------------------------------------------------------
//
void CalibrationTask::update() {
  LogDebug("Commissioning") << "[CalibrationTask::update]";
  for(std::vector<HistoSet>::iterator it=calib_.begin();it<calib_.end();++it) {
    updateHistoSet( *it );
  }
}

// -----------------------------------------------------------------------------
//
void CalibrationTask::checkAndSave(const uint16_t& calchan) {
  //for the first time
  if(lastCalChan_==0) {
    // set the title
    for(int apv=0;apv<2;++apv) {
      for(int i=0;i<16;++i) {
        std::stringstream complement;
	int isub = calchan<4 ? calchan+4 : calchan-4;
        complement << "STRIP_" << (isub+i*8) ;
        std::string title = SiStripHistoTitle( sistrip::EXPERT_HISTO, 
                                               runType_, 
                                               sistrip::DET_KEY, 
                                               connection().detId(),
                                               sistrip::APV, 
                                               connection().i2cAddr(apv),
                                               complement.str() ).title(); 
        calib_[apv*16+i].histo_->setTitle(title);
      }
    }

    // set new parameter value
    lastCalChan_ = calchan;
    return;
  }
  //check if ISHA or VFS has changed
  if(calchan!=lastCalChan_) {
    // in that case, save the histograms, reset them, change the title and continue
    
    // set histograms before saving
    update();

    // Strip filename of ".root" extension
    std::string name;
    if ( filename_.find(".root",0) == std::string::npos ) { name = filename_; }
    else { name = filename_.substr( 0, filename_.find(".root",0) ); }
  
    // Retrieve SCRATCH directory
    std::string scratch = "SCRATCH"; //@@ remove trailing slash!!!
    std::string dir = "";
    if ( getenv(scratch.c_str()) != NULL ) {
      dir = getenv(scratch.c_str());
    }
  
    // Save file with appropriate filename 
    std::stringstream ss;
    if ( !dir.empty() ) { ss << dir << "/"; }
    else { ss << "/tmp/"; }
    ss << "_CALCHAN" << lastCalChan_; // Add CalChan value
    ss << "_000"; // Add FU instance number (fake)
    ss << ".root"; // Append ".root" extension
    if ( !filename_.empty() ) {
      dqm()->save( ss.str() ); 
      LogTrace("DQMsource")
        << "[SiStripCommissioningSource::" << __func__ << "]"
        << " Saved all histograms to file \""
        << ss.str() << "\"";
    } else {
      edm::LogWarning("DQMsource")
        << "[SiStripCommissioningSource::" << __func__ << "]"
        << " NULL value for filename! No root file saved!";
    }
  
    // reset
    for(std::vector<HistoSet>::iterator it=calib_.begin();it<calib_.end();++it) {
      it->vNumOfEntries_.clear();
      it->vNumOfEntries_.resize(nBins_,0);
    }
  
    // change the title
    for(int apv=0;apv<2;++apv) {
      for(int i=0;i<16;++i) {
        std::stringstream complement;
	int isub = calchan<4 ? calchan+4 : calchan-4;
        complement << "STRIP_" << (isub+i*8) ;
        std::string title = SiStripHistoTitle( sistrip::EXPERT_HISTO, 
                                               runType_, 
                                               sistrip::DET_KEY, 
                                               connection().detId(),
                                               sistrip::APV, 
                                               connection().i2cAddr(apv),
                                               complement.str() ).title(); 
        calib_[apv*16+i].histo_->setTitle(title);
      }
    }

    // set new parameter values
    lastCalChan_ = calchan;
  }
}


#include "DQM/SiStripCommissioningSources/interface/CalibrationScanTask.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// -----------------------------------------------------------------------------
//
CalibrationScanTask::CalibrationScanTask( DaqMonitorBEInterface* dqm,
			      const FedChannelConnection& conn,
			      const sistrip::RunType& rtype,
			      const char* filename ) :
  CommissioningTask( dqm, conn, "CalibrationScanTask" ),
  runType_(rtype),
  calib1_(),calib2_(),
  nBins_(65),lastISHA_(0),lastVFS_(0),
  filename_(filename)
{
  LogDebug("Commissioning") << "[CalibrationScanTask::CalibrationScanTask] Constructing object...";
}

// -----------------------------------------------------------------------------
//
CalibrationScanTask::~CalibrationScanTask() {
  LogDebug("Commissioning") << "[CalibrationScanTask::CalibrationScanTask] Destructing object...";
}

// -----------------------------------------------------------------------------
//
void CalibrationScanTask::book() {
  LogDebug("Commissioning") << "[CalibrationScanTask::book]";

  // construct the histo titles and book two histograms: one per APV
  std::string title1 = SiStripHistoTitle( sistrip::EXPERT_HISTO, 
					 runType_, 
  					 sistrip::DET_KEY, 
					 connection().detId(),
					 sistrip::APV, 
					 connection().i2cAddr(0) ).title(); 
  calib1_.histo_ = dqm()->book1D( title1, title1, nBins_, 0, 203.125);
  calib1_.isProfile_=false;
  calib1_.vNumOfEntries_.resize(nBins_,0);
//  calib1_.vSumOfContents_.resize(nBins_,0);
//  calib1_.vSumOfSquares_.resize(nBins_,0);
  std::string title2 = SiStripHistoTitle( sistrip::EXPERT_HISTO, 
					 runType_, 
  					 sistrip::DET_KEY, 
					 connection().detId(),
					 sistrip::APV, 
					 connection().i2cAddr(1) ).title(); 
  calib2_.histo_ = dqm()->book1D( title2, title2, nBins_, 0, 203.125);
  calib2_.isProfile_=false;
  calib2_.vNumOfEntries_.resize(nBins_,0);
//  calib2_.vSumOfContents_.resize(nBins_,0);
//  calib2_.vSumOfSquares_.resize(nBins_,0);
  LogDebug("Commissioning") << "[CalibrationScanTask::book] done";
  
}

// -----------------------------------------------------------------------------
//
void CalibrationScanTask::fill( const SiStripEventSummary& summary,
			    const edm::DetSet<SiStripRawDigi>& digis ) {
  LogDebug("Commissioning") << "[CalibrationScanTask::fill]";
  // Check if ISHA/VFS changed. In that case, save, reset histo, change title, and continue
  checkAndSave(summary.isha(),summary.vfs());
  // retrieve the delay from the EventSummary
  int bin = (100-summary.latency())*8+(7-summary.calSel());
  if(runType_==sistrip::CALIBRATION_SCAN) bin +=16; // difference between PEAK and DECO
  // loop on the strips to fill the histogram
  // digis are obtained for an APV pair. 
  // strips 0->127  : calib1_
  // strips 128->255: calib2_
  // then, only some strips are fired at a time.
  // We use calChan to know that
  int isub,ical = summary.calChan();
  isub = ical<4 ? ical+4 : ical-4;
  for (int k=ical;k<128;k+=8)  {
  //TODO: we must substract the pedestal (simplest would be to read it when constructing the task)
    // all strips of the APV are merged in 
    updateHistoSet( calib1_,bin,digis.data[k].adc());
    updateHistoSet( calib2_,bin,digis.data[k+128].adc());
  }

}

// -----------------------------------------------------------------------------
//
void CalibrationScanTask::update() {
  LogDebug("Commissioning") << "[CalibrationScanTask::update]";
  updateHistoSet( calib1_ );
  updateHistoSet( calib2_ );
}

// -----------------------------------------------------------------------------
//
void CalibrationScanTask::checkAndSave(const uint16_t& isha, const uint16_t& vfs ) {
  //for the first time
  if(lastISHA_==0 && lastVFS_==0) {
    // set the title
    std::stringstream complement;
    complement << "ISHA" << isha << "_VFS" << vfs;
    std::string title1 = SiStripHistoTitle( sistrip::EXPERT_HISTO, 
			 		    runType_, 
  					    sistrip::DET_KEY, 
				 	    connection().detId(),
					    sistrip::APV, 
					    connection().i2cAddr(0),
					    complement.str() ).title(); 
    calib1_.histo_->setTitle(title1);

    std::string title2 = SiStripHistoTitle( sistrip::EXPERT_HISTO, 
			 		    runType_, 
  					    sistrip::DET_KEY, 
				 	    connection().detId(),
					    sistrip::APV, 
					    connection().i2cAddr(1),
					    complement.str() ).title(); 
    calib2_.histo_->setTitle(title2);

    // set new parameter values
    lastISHA_=isha;
    lastVFS_=vfs;
    return;
  }
  //check if ISHA or VFS has changed
  if(isha!=lastISHA_ || vfs!=lastVFS_) {
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
    ss << "_ISHA" << lastISHA_ << "_VFS" << lastVFS_; // Add ISHA and VFS values
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
    calib1_.vNumOfEntries_.clear();
    calib1_.vNumOfEntries_.resize(nBins_,0);
    calib2_.vNumOfEntries_.clear();
    calib2_.vNumOfEntries_.resize(nBins_,0);
  
    // change the title
    std::stringstream complement;
    complement << "ISHA" << isha << "_VFS" << vfs;
    std::string title1 = SiStripHistoTitle( sistrip::EXPERT_HISTO, 
			 		    runType_, 
  					    sistrip::DET_KEY, 
				 	    connection().detId(),
					    sistrip::APV, 
					    connection().i2cAddr(0),
					    complement.str() ).title(); 
    calib1_.histo_->setTitle(title1);

    std::string title2 = SiStripHistoTitle( sistrip::EXPERT_HISTO, 
			 		    runType_, 
  					    sistrip::DET_KEY, 
				 	    connection().detId(),
					    sistrip::APV, 
					    connection().i2cAddr(1),
					    complement.str() ).title(); 
    calib2_.histo_->setTitle(title2);

    // set new parameter values
    lastISHA_=isha;
    lastVFS_=vfs;
  }
}


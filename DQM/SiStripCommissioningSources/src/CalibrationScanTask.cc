#include "DQM/SiStripCommissioningSources/interface/CalibrationScanTask.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include <CondFormats/DataRecord/interface/SiStripPedestalsRcd.h>
#include <CondFormats/SiStripObjects/interface/SiStripPedestals.h>
#include <DQMServices/Core/interface/MonitorElement.h>

#include <arpa/inet.h>
#include <sys/unistd.h>
#include <sys/socket.h>
#include <netdb.h>
#include <stdio.h>
#include <fstream>

// -----------------------------------------------------------------------------
//
CalibrationScanTask::CalibrationScanTask( DQMStore* dqm,
			      const FedChannelConnection& conn,
			      const sistrip::RunType& rtype,
			      const char* filename,
			      uint32_t run,
                              const edm::EventSetup& setup ) :
  CommissioningTask( dqm, conn, "CalibrationScanTask" ),
  runType_(rtype),
  calib1_(),calib2_(),
  nBins_(65),lastISHA_(1000),lastVFS_(1000),lastCalchan_(1000),
  filename_(filename),
  run_(run)
{
  LogDebug("Commissioning") << "[CalibrationScanTask::CalibrationScanTask] Constructing object...";
  // load the pedestals
  edm::ESHandle<SiStripPedestals> pedestalsHandle;
  setup.get<SiStripPedestalsRcd>().get(pedestalsHandle);
  SiStripPedestals::Range detPedRange = pedestalsHandle->getRange(conn.detId());
  int start = conn.apvPairNumber()*256;
  int stop  = start + 256;
  int value = 0;
  ped.reserve(256);
  LogDebug("Commissioning") << "[CalibrationScanTask::CalibrationScanTask] Loading pedestal for " << conn.detId();
  if(conn.detId()==0) return;
  for(int strip = start; strip < stop; ++strip) {
    value = int(pedestalsHandle->getPed(strip,detPedRange));
    if(value>895) value -= 1024;
    ped.push_back(value);
  }
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
  calib1_.histo( dqm()->book1D( title1, title1, nBins_, 0, 203.125) );
  calib1_.isProfile_=false;
  calib1_.vNumOfEntries_.resize(nBins_,0);
  std::string title2 = SiStripHistoTitle( sistrip::EXPERT_HISTO, 
					 runType_, 
  					 sistrip::DET_KEY, 
					 connection().detId(),
					 sistrip::APV, 
					 connection().i2cAddr(1) ).title(); 
  calib2_.histo( dqm()->book1D( title2, title2, nBins_, 0, 203.125) );
  calib2_.isProfile_=false;
  calib2_.vNumOfEntries_.resize(nBins_,0);

  // book the isha, vfs values
  std::string pwd = dqm()->pwd();
  std::string rootDir = pwd.substr(0,pwd.find(std::string(sistrip::root_) + "/")+(sizeof(sistrip::root_) - 1));
  dqm()->setCurrentFolder( rootDir );
  std::vector<std::string> existingMEs = dqm()->getMEs();
  if(find(existingMEs.begin(),existingMEs.end(),"isha")!=existingMEs.end()) {
    ishaElement_ = dqm()->get(dqm()->pwd()+"/isha");
  } else {
    ishaElement_ = dqm()->bookInt("isha");
  }
  if(find(existingMEs.begin(),existingMEs.end(),"isha")!=existingMEs.end()) {
    vfsElement_ = dqm()->get(dqm()->pwd()+"/vfs");  
  } else {
    vfsElement_ = dqm()->bookInt("vfs");
  }
  if(find(existingMEs.begin(),existingMEs.end(),"calchan")!=existingMEs.end()) {
    calchanElement_ = dqm()->get(dqm()->pwd()+"/calchan");
  } else {
    calchanElement_ = dqm()->bookInt("calchan");
  }
  
  LogDebug("Commissioning") << "[CalibrationScanTask::book] done";
  
}

// -----------------------------------------------------------------------------
//
void CalibrationScanTask::fill( const SiStripEventSummary& summary,
			    const edm::DetSet<SiStripRawDigi>& digis ) {
//  LogDebug("Commissioning") << "[CalibrationScanTask::fill]: isha/vfs = " << summary.isha() << "/" << summary.vfs();
  // Check if ISHA/VFS changed. In that case, save, reset histo, change title, and continue
  checkAndSave(summary.isha(),summary.vfs());
  // retrieve the delay from the EventSummary
  int bin = (100-summary.latency())*8+(7-summary.calSel());
  // loop on the strips to fill the histogram
  // digis are obtained for an APV pair. 
  // strips 0->127  : calib1_
  // strips 128->255: calib2_
  // then, only some strips are fired at a time.
  // We use calChan to know that
  int isub,ical = summary.calChan();
  isub = ical<4 ? ical+4 : ical-4;
  lastCalchan_ = ical;
  for (int k=0;k<16;++k)  {
    // all strips of the APV are merged in 
    updateHistoSet( calib1_,bin,digis.data[ical+k*8].adc()-ped[ical+k*8]-(digis.data[isub+k*8].adc()-ped[isub+k*8]));
    updateHistoSet( calib2_,bin,digis.data[128+ical+k*8].adc()-ped[128+ical+k*8]-(digis.data[128+isub+k*8].adc()-ped[128+isub+k*8]));
  }
  update(); //TODO: temporary: find a better solution later
}

// -----------------------------------------------------------------------------
//
void CalibrationScanTask::update() {
  // LogDebug("Commissioning") << "[CalibrationScanTask::update]"; // huge output
  updateHistoSet( calib1_ );
  updateHistoSet( calib2_ );
}

// -----------------------------------------------------------------------------
//
void CalibrationScanTask::checkAndSave(const uint16_t& isha, const uint16_t& vfs ) {
  //for the first time
  if(lastISHA_==1000 && lastVFS_==1000) {
    lastISHA_ = isha;
    lastVFS_  = vfs;
  }

  // set the calchan value in the correspond string
  ishaElement_->Fill(lastISHA_);
  vfsElement_->Fill(lastVFS_);
  calchanElement_->Fill(lastCalchan_);

  // check if ISHA/VFS has changed
  // in that case, save the histograms, reset them, change the title and continue
  if(lastISHA_!=isha || lastVFS_!=vfs) {

    // set histograms before saving
    update(); //TODO: we must do this for all tasks, otherwise only the first is updated.
    
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
    calib1_.histo()->setTitle(title1);

    std::string title2 = SiStripHistoTitle( sistrip::EXPERT_HISTO, 
			 		    runType_, 
  					    sistrip::DET_KEY, 
				 	    connection().detId(),
					    sistrip::APV, 
					    connection().i2cAddr(1),
					    complement.str() ).title(); 
    calib2_.histo()->setTitle(title2);

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
    ss << name << "_";
    directory(ss,run_); // Add filename with run number, host ip, pid and .root extension
    ss << "_ISHA" << lastISHA_ << "_VFS" << lastVFS_; // Add ISHA and VFS values
    ss << "_000"; // Add FU instance number (fake)
    ss << ".root"; // Append ".root" extension
    if ( !filename_.empty() ) {
      if(ifstream(ss.str().c_str(),ifstream::in).fail()) { //save only once. Skip if the file already exist
        dqm()->save( ss.str() ); 
        LogTrace("DQMsource")
          << "[SiStripCommissioningSource::" << __func__ << "]"
          << " Saved all histograms to file \""
          << ss.str() << "\"";
      } else {
        LogTrace("DQMsource")
	  << "[SiStripCommissioningSource::" << __func__ << "]"
	  << " Skipping creation of file \""
	  << ss.str() << "\" that already exists" ;
      }
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
  
    // set new parameter values
    lastISHA_=isha;
    lastVFS_=vfs;
    // set the calchan value in the correspond string
    ishaElement_->Fill(lastISHA_);
    vfsElement_->Fill(lastVFS_);
    calchanElement_->Fill(lastCalchan_);
  }
}

// -----------------------------------------------------------------------------
// 
void CalibrationScanTask::directory( std::stringstream& dir,
					    uint32_t run_number ) {

  // Get details about host
  char hn[256];
  gethostname( hn, sizeof(hn) );
  struct hostent* he;
  he = gethostbyname(hn);

  // Extract host name and ip
  std::string host_name;
  std::string host_ip;
  if ( he ) { 
    host_name = std::string(he->h_name);
    host_ip = std::string( inet_ntoa( *(struct in_addr*)(he->h_addr) ) );
  } else {
    host_name = "unknown.cern.ch";
    host_ip = "255.255.255.255";
  }

  // Reformat IP address
  std::string::size_type pos = 0;
  std::stringstream ip;
  //for ( uint16_t ii = 0; ii < 4; ++ii ) {
  while ( pos != std::string::npos ) {
    std::string::size_type tmp = host_ip.find(".",pos);
    if ( tmp != std::string::npos ) {
      ip << std::setw(3)
	 << std::setfill('0')
	 << host_ip.substr( pos, tmp-pos ) 
	 << ".";
      pos = tmp+1; // skip the delimiter "."
    } else {
      ip << std::setw(3)
	 << std::setfill('0')
	 << host_ip.substr( pos );
      pos = std::string::npos;
    }
  }
  
  // Get pid
  pid_t pid = getpid();

  // Construct string
  dir << std::setw(8) 
      << std::setfill('0') 
      << run_number
      << "_" 
      << ip.str()
      << "_"
      << std::setw(5) 
      << std::setfill('0') 
      << pid;

}

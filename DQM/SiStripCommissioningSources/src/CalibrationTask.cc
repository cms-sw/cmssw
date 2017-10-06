#include "DQM/SiStripCommissioningSources/interface/CalibrationTask.h"
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
#include <cstdio>
#include <fstream>

// -----------------------------------------------------------------------------
//
CalibrationTask::CalibrationTask( DQMStore* dqm,
			      const FedChannelConnection& conn,
			      const sistrip::RunType& rtype,
			      const char* filename,
			      uint32_t run,
			      const edm::EventSetup& setup) :
  CommissioningTask( dqm, conn, "CalibrationTask" ),
  runType_(rtype),
  nBins_(65),lastCalChan_(8),
  filename_(filename),
  run_(run)
{
  LogDebug("Commissioning") << "[CalibrationTask::CalibrationTask] Constructing object...";
  // load the pedestals
  edm::ESHandle<SiStripPedestals> pedestalsHandle;
  setup.get<SiStripPedestalsRcd>().get(pedestalsHandle);
  SiStripPedestals::Range detPedRange = pedestalsHandle->getRange(conn.detId());
  int start = conn.apvPairNumber()*256; 
  int stop  = start + 256;
  int value = 0;
  ped.reserve(256);
  for(int strip = start; strip < stop; ++strip) {
    value = int(pedestalsHandle->getPed(strip,detPedRange));
    if(value>895) value -= 1024;
    ped.push_back(value);
  }
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
      std::stringstream complement;
      complement << "_" << i ;
      title += complement.str();			
      calib_[apv*16+i].histo( dqm()->book1D( title, title, nBins_, 0, 203.125) );
      calib_[apv*16+i].isProfile_=false;
      calib_[apv*16+i].vNumOfEntries_.resize(nBins_,0);
    }
  }

  // book the calchan values
  std::string pwd = dqm()->pwd();
  std::string rootDir = pwd.substr(0,pwd.find(std::string(sistrip::root_) + "/")+(sizeof(sistrip::root_) - 1));
  dqm()->setCurrentFolder( rootDir );
  std::vector<std::string> existingMEs = dqm()->getMEs();
  if(find(existingMEs.begin(),existingMEs.end(),"calchan")!=existingMEs.end()) {
    calchanElement_ = dqm()->get(dqm()->pwd()+"/calchan");
  } else {
    calchanElement_ = dqm()->bookInt("calchan");
  }
  dqm()->setCurrentFolder(pwd);
  
  LogDebug("Commissioning") << "[CalibrationTask::book] done";
  
}

// -----------------------------------------------------------------------------
//
void CalibrationTask::fill( const SiStripEventSummary& summary,
			    const edm::DetSet<SiStripRawDigi>& digis ) {
  LogDebug("Commissioning") << "[CalibrationTask::fill]";
  // Check if CalChan changed. In that case, save, reset histo, change title, and continue
  int isub,ical = summary.calChan();
  isub = ical<4 ? ical+4 : ical-4;
  checkAndSave(ical);
  // retrieve the delay from the EventSummary
  int bin = (100-summary.latency())*8+(7-summary.calSel());
  // Fill the histograms.
  // data-ped -(data-ped)_isub
  // the second term corresponds to the common mode substraction, looking at a strip far away.
  for (int apv=0; apv<2; ++apv) {
    for (int k=0;k<16;++k)  {
      updateHistoSet( calib_[apv*16+k],bin,digis.data[apv*128+ical+k*8].adc()-ped[apv*128+ical+k*8]-(digis.data[apv*128+isub+k*8].adc()-ped[apv*128+isub+k*8]));
    }
  }
  update(); //TODO: temporary: find a better solution later
}

// -----------------------------------------------------------------------------
//
void CalibrationTask::update() {
  // LogDebug("Commissioning") << "[CalibrationTask::update]"; // huge output
  for(std::vector<HistoSet>::iterator it=calib_.begin();it<calib_.end();++it) {
    updateHistoSet( *it );
  }
}

// -----------------------------------------------------------------------------
//
void CalibrationTask::checkAndSave(const uint16_t& calchan) {
  //for the first time
  if(lastCalChan_==8) {
    // set new parameter value
    lastCalChan_ = calchan;
    calchanElement_->Fill(lastCalChan_);
    return;
  }

  //check if CDRV register (ical) has changed
  // in that case, save the histograms, reset them, change the title and continue
  if(calchan!=lastCalChan_) {
    
    // set histograms before saving
    update(); //TODO: we must do this for all tasks, otherwise only the first is updated.

    // change the title
    for(int apv=0;apv<2;++apv) {
      for(int i=0;i<16;++i) {
        std::stringstream complement;
        complement << "STRIP_" << (apv*128+lastCalChan_+i*8) ;
        std::string title = SiStripHistoTitle( sistrip::EXPERT_HISTO, 
                                               runType_, 
                                               sistrip::DET_KEY, 
                                               connection().detId(),
                                               sistrip::APV, 
                                               connection().i2cAddr(apv),
                                               complement.str() ).title(); 
        calib_[apv*16+i].histo()->setTitle(title);
	calib_[apv*16+i].histo()->setAxisTitle(title);
      }
    }

    // Strip filename of ".root" extension
    std::string name;
    if ( filename_.find(".root",0) == std::string::npos ) { name = filename_; }
    else { name = filename_.substr( 0, filename_.find(".root",0) ); }
  
    // Retrieve SCRATCH directory
    std::string scratch = "SCRATCH"; //@@ remove trailing slash!!!
    std::string dir = "";
    if ( getenv(scratch.c_str()) != nullptr ) {
      dir = getenv(scratch.c_str());
    }
  
    // Save file with appropriate filename 
    std::stringstream ss;
    if ( !dir.empty() ) { ss << dir << "/"; }
    else { ss << "/tmp/"; }
    ss << name << "_";
    directory(ss,run_); // Add filename with run number, host ip, pid and .root extension
    ss << "_CALCHAN" << lastCalChan_; // Add CalChan value
    ss << "_000"; // Add FU instance number (fake)
    ss << ".root"; // Append ".root" extension
    if ( !filename_.empty() ) {
      if(std::ifstream(ss.str().c_str(),std::ifstream::in).fail()) { //save only once. Skip if the file already exist
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
    for(std::vector<HistoSet>::iterator it=calib_.begin();it<calib_.end();++it) {
      it->vNumOfEntries_.clear();
      it->vNumOfEntries_.resize(nBins_,0);
    }
    // set new parameter values
    lastCalChan_ = calchan;
    calchanElement_->Fill(lastCalChan_);
  }
}

// -----------------------------------------------------------------------------
// 
void CalibrationTask::directory( std::stringstream& dir,
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

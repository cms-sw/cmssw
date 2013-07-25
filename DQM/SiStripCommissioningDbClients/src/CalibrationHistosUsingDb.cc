// Last commit: $Id: CalibrationHistosUsingDb.cc,v 1.12 2009/11/10 14:49:02 lowette Exp $

#include "DQM/SiStripCommissioningDbClients/interface/CalibrationHistosUsingDb.h"
#include "CondFormats/SiStripObjects/interface/CalibrationAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include "DQM/SiStripCommon/interface/ExtractTObject.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

using namespace sistrip;

// -----------------------------------------------------------------------------
/** */
std::string getBasePath (const std::string &path)
{
  return path.substr(0,path.find(std::string(sistrip::root_) + "/")+sizeof(sistrip::root_) );
}

// -----------------------------------------------------------------------------
/** */
CalibrationHistosUsingDb::CalibrationHistosUsingDb( const edm::ParameterSet & pset,
                                                    DQMStore* bei,
                                                    SiStripConfigDb* const db,
                                                    const sistrip::RunType& task ) 
  : CommissioningHistograms( pset.getParameter<edm::ParameterSet>("CalibrationParameters"),
                             bei,
                             task ),
    CommissioningHistosUsingDb( db,
                                task ),
    CalibrationHistograms( pset.getParameter<edm::ParameterSet>("CalibrationParameters"),
                           bei,
                           task )
{
  LogTrace(mlDqmClient_) 
    << "[CalibrationHistosUsingDb::" << __func__ << "]"
    << " Constructing object...";
  // Load and dump the current ISHA/VFS values. This is used by the standalone analysis script
  const SiStripConfigDb::DeviceDescriptionsRange & apvDescriptions = db->getDeviceDescriptions(APV25);
  for(SiStripConfigDb::DeviceDescriptionsV::const_iterator apv = apvDescriptions.begin();apv!=apvDescriptions.end();++apv) {
    apvDescription* desc = dynamic_cast<apvDescription*>( *apv );
    if ( !desc ) { continue; }
    // Retrieve device addresses from device description
    const SiStripConfigDb::DeviceAddress& addr = db->deviceAddress(*desc);
    std::stringstream bin;
    bin        << std::setw(1) << std::setfill('0') << addr.fecCrate_;
    bin << "." << std::setw(2) << std::setfill('0') << addr.fecSlot_;
    bin << "." << std::setw(1) << std::setfill('0') << addr.fecRing_;
    bin << "." << std::setw(3) << std::setfill('0') << addr.ccuAddr_;
    bin << "." << std::setw(2) << std::setfill('0') << addr.ccuChan_;
    bin << "." << desc->getAddress();
    LogTrace(mlDqmClient_) << "Present values for ISHA/VFS of APV " 
			   << bin.str() << " : " 
			   << static_cast<uint16_t>(desc->getIsha()) << " " << static_cast<uint16_t>(desc->getVfs());
  }
  // Load the histograms with the results
  std::string pwd = bei->pwd();
  std::string ishaPath = getBasePath(pwd);
  ishaPath += "/ControlView/isha";
  LogTrace(mlDqmClient_) << "Looking for " << ishaPath;
  ishaHistogram_ = ExtractTObject<TH1F>().extract( bei->get(ishaPath) );
  std::string vfsPath = getBasePath(pwd);
  vfsPath += "/ControlView/vfs";
  LogTrace(mlDqmClient_) << "Looking for " << vfsPath;
  vfsHistogram_ = ExtractTObject<TH1F>().extract( bei->get(vfsPath) );
  
}

// -----------------------------------------------------------------------------
/** */
CalibrationHistosUsingDb::~CalibrationHistosUsingDb() {
  LogTrace(mlDqmClient_) 
    << "[CalibrationHistosUsingDb::" << __func__ << "]"
    << " Destructing object...";
}

// -----------------------------------------------------------------------------
/** */
void CalibrationHistosUsingDb::uploadConfigurations() {
  
  LogTrace(mlDqmClient_)
    << "[CalibrationHistosUsingDb::" << __func__ << "]" << ishaHistogram_ << " " << vfsHistogram_;

  if(!ishaHistogram_ && !vfsHistogram_) return;

  if ( !db() ) {
    edm::LogWarning(mlDqmClient_) 
      << "[CalibrationHistosUsingDb::" << __func__ << "]"
      << " NULL pointer to SiStripConfigDb interface!"
      << " Aborting upload...";
    return;
  }
  
  // Update all APV device descriptions with new ISHA and VFS settings
  SiStripConfigDb::DeviceDescriptionsRange devices = db()->getDeviceDescriptions();
  update( devices );
  if ( doUploadConf() ) {
    edm::LogVerbatim(mlDqmClient_)
      << "[CalibrationHistosUsingDb::" << __func__ << "]"
      << " Uploading ISHA/VFS settings to DB...";
    db()->uploadDeviceDescriptions();
    edm::LogVerbatim(mlDqmClient_)
      << "[CalibrationHistosUsingDb::" << __func__ << "]"
      << " Uploaded ISHA/VFS settings to DB!";
  } else {
    edm::LogWarning(mlDqmClient_)
      << "[CalibrationHistosUsingDb::" << __func__ << "]"
      << " TEST only! No ISHA/VFS settings will be uploaded to DB...";
  }

  LogTrace(mlDqmClient_)
    << "[CalibrationHistosUsingDb::" << __func__ << "]"
    << " Upload of ISHA/VFS settings to DB finished!";

}

// -----------------------------------------------------------------------------
/** */
void CalibrationHistosUsingDb::update( SiStripConfigDb::DeviceDescriptionsRange& devices ) {
  
  if(!ishaHistogram_ && !vfsHistogram_) return;
  
  // Iterate through devices and update device descriptions
  SiStripConfigDb::DeviceDescriptionsV::const_iterator idevice;
  for ( idevice = devices.begin(); idevice != devices.end(); idevice++ ) {

    // Check device type
    if ( (*idevice)->getDeviceType() != APV25 ) { continue; }

    // Cast to retrieve appropriate description object
    apvDescription* desc = dynamic_cast<apvDescription*>( *idevice );
    if ( !desc ) { continue; }

    // Retrieve the device address from device description
    const SiStripConfigDb::DeviceAddress& addr = db()->deviceAddress(*desc);

    // Construct the string for that address
    std::stringstream bin;
    bin        << std::setw(1) << std::setfill('0') << addr.fecCrate_;
    bin << "." << std::setw(2) << std::setfill('0') << addr.fecSlot_;
    bin << "." << std::setw(1) << std::setfill('0') << addr.fecRing_;
    bin << "." << std::setw(3) << std::setfill('0') << addr.ccuAddr_;
    bin << "." << std::setw(2) << std::setfill('0') << addr.ccuChan_;
    bin << "." << desc->getAddress();

    // Iterate over the histo bins and find the right one
    for(int i = 1;i <= ishaHistogram_->GetNbinsX(); ++i) {
      std::string label = ishaHistogram_->GetXaxis()->GetBinLabel(i);
      if(label == bin.str()) {
        desc->setIsha( (int)round(ishaHistogram_->GetBinContent(i)) );
	LogDebug(mlDqmClient_) << "Setting ISHA to " << ((int)round(ishaHistogram_->GetBinContent(i))) << " for " << label;
      }
    }
    for(int i = 1;i <= vfsHistogram_->GetNbinsX(); ++i) {
      std::string label = vfsHistogram_->GetXaxis()->GetBinLabel(i);
      if(label == bin.str()) {
        desc->setVfs( (int)round(vfsHistogram_->GetBinContent(i)) );
	LogDebug(mlDqmClient_) << "Setting VFS to " << ((int)round(vfsHistogram_->GetBinContent(i))) << " for " << label;
      }
    }
    
  }

}

// -----------------------------------------------------------------------------
/** */
void CalibrationHistosUsingDb::create( SiStripConfigDb::AnalysisDescriptionsV& desc,
				       Analysis analysis) {

  CalibrationAnalysis* anal = dynamic_cast<CalibrationAnalysis*>( analysis->second );
  if ( !anal ) { return; }

  SiStripFecKey fec_key( anal->fecKey() );
  SiStripFedKey fed_key( anal->fedKey() );

  std::ofstream ofile("calibrationResults.txt",ios_base::app);
  for ( uint16_t iapv = 0; iapv < 2; ++iapv ) {

    // Create description
    CalibrationAnalysisDescription *tmp;
    tmp = new CalibrationAnalysisDescription(anal->amplitudeMean()[iapv],
                                             anal->tailMean()[iapv],
  					     anal->riseTimeMean()[iapv],
  					     anal->timeConstantMean()[iapv],
  					     anal->smearingMean()[iapv],
  					     anal->chi2Mean()[iapv],
  					     anal->deconvMode(),
  					     fec_key.fecCrate(),
  					     fec_key.fecSlot(),
  					     fec_key.fecRing(),
  					     fec_key.ccuAddr(),
  					     fec_key.ccuChan(),
  					     SiStripFecKey::i2cAddr( fec_key.lldChan(), !iapv ),
  					     db()->dbParams().partitions().begin()->second.partitionName(),
  					     db()->dbParams().partitions().begin()->second.runNumber(),
  					     anal->isValid(),
  					     "",
  					     fed_key.fedId(),
  					     fed_key.feUnit(),
  					     fed_key.feChan(),
  					     fed_key.fedApv(),
					     calchan_,
					     isha_,
					     vfs_ );
  
    // debug simplified printout in text file
    ofile << " " <<  anal->amplitudeMean()[iapv]
          << " " <<  anal->tailMean()[iapv]
          << " " <<  anal->riseTimeMean()[iapv]
          << " " <<  anal->timeConstantMean()[iapv]
          << " " <<  anal->smearingMean()[iapv]
          << " " <<  anal->chi2Mean()[iapv]
          << " " <<  anal->deconvMode()
          << " " <<  fec_key.fecCrate()
          << " " <<  fec_key.fecSlot()
          << " " <<  fec_key.fecRing()
          << " " <<  fec_key.ccuAddr()
          << " " <<  fec_key.ccuChan()
          << " " <<  SiStripFecKey::i2cAddr( fec_key.lldChan(), !iapv )
          << " " <<  db()->dbParams().partitions().begin()->second.partitionName()
          << " " <<  db()->dbParams().partitions().begin()->second.runNumber()
          << " " <<  fed_key.fedId()
          << " " <<  fed_key.feUnit()
          << " " <<  fed_key.feChan()
          << " " <<  fed_key.fedApv()
          << " " <<  calchan_
          << " " <<  isha_
          << " " <<  vfs_ << std::endl;

    // Add comments
    typedef std::vector<std::string> Strings;
    Strings errors = anal->getErrorCodes();
    Strings::const_iterator istr = errors.begin();
    Strings::const_iterator jstr = errors.end();
    for ( ; istr != jstr; ++istr ) { tmp->addComments( *istr ); }
  
    // Store description
    desc.push_back( tmp );
  }
  ofile.close();

}


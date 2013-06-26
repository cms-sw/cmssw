/*
 * \file EcalEndcapMonitorClient.cc
 *
 * $Date: 2013/04/02 09:03:29 $
 * $Revision: 1.274 $
 * \author G. Della Ricca
 * \author F. Cossutti
 *
*/

#include <memory>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <unistd.h>

#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/EcalRawData/interface/EcalDCCHeaderBlock.h"

#ifdef WITH_ECAL_COND_DB
#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"
#include "OnlineDB/EcalCondDB/interface/RunTag.h"
#include "OnlineDB/EcalCondDB/interface/RunDat.h"
#include "OnlineDB/EcalCondDB/interface/MonRunDat.h"
#include "DQM/EcalCommon/interface/LogicID.h"
#endif

#include "DQM/EcalCommon/interface/Masks.h"

#include "DQM/EcalCommon/interface/UtilsClient.h"
#include "DQM/EcalCommon/interface/Numbers.h"

#include "DQM/EcalEndcapMonitorClient/interface/EcalEndcapMonitorClient.h"

#include "DQM/EcalEndcapMonitorClient/interface/EEIntegrityClient.h"
#include "DQM/EcalEndcapMonitorClient/interface/EEStatusFlagsClient.h"
#include "DQM/EcalEndcapMonitorClient/interface/EEOccupancyClient.h"
#include "DQM/EcalEndcapMonitorClient/interface/EECosmicClient.h"
#include "DQM/EcalEndcapMonitorClient/interface/EELaserClient.h"
#include "DQM/EcalEndcapMonitorClient/interface/EEPedestalClient.h"
#include "DQM/EcalEndcapMonitorClient/interface/EEPedestalOnlineClient.h"
#include "DQM/EcalEndcapMonitorClient/interface/EETestPulseClient.h"
#include "DQM/EcalEndcapMonitorClient/interface/EETriggerTowerClient.h"
#include "DQM/EcalEndcapMonitorClient/interface/EEClusterClient.h"
#include "DQM/EcalEndcapMonitorClient/interface/EETimingClient.h"
#include "DQM/EcalEndcapMonitorClient/interface/EELedClient.h"

EcalEndcapMonitorClient::EcalEndcapMonitorClient(const edm::ParameterSet& ps) 
{
  // verbose switch

  verbose_ = ps.getUntrackedParameter<bool>("verbose",  true);

  if ( verbose_ ) {
    std::cout << std::endl;
    std::cout << " *** Ecal Endcap Generic Monitor Client ***" << std::endl;
    std::cout << std::endl;
  }

  // DQM ROOT input file

  inputFile_ = ps.getUntrackedParameter<std::string>("inputFile", "");

  if ( verbose_ ) {
    if ( inputFile_.size() != 0 ) {
      std::cout << " Reading DQM data from inputFile '" << inputFile_ << "'" << std::endl;
    }
  }

  // Ecal Cond DB

  dbName_ = ps.getUntrackedParameter<std::string>("dbName", "");
  dbHostName_ = ps.getUntrackedParameter<std::string>("dbHostName", "");
  dbHostPort_ = ps.getUntrackedParameter<int>("dbHostPort", 1521);
  dbUserName_ = ps.getUntrackedParameter<std::string>("dbUserName", "");
  dbPassword_ = ps.getUntrackedParameter<std::string>("dbPassword", "");

  dbTagName_ = ps.getUntrackedParameter<std::string>("dbTagName", "CMSSW");

  if ( verbose_ ) {
    if ( dbName_.size() != 0 ) {
      std::cout << " Ecal Cond DB: " << std::endl;
      std::cout << "   dbName = '" << dbName_ << "'" << std::endl;
      std::cout << "   dbUserName = '" << dbUserName_ << "'" << std::endl;
      if ( dbHostName_.size() != 0 ) {
        std::cout << "   dbHostName = '" << dbHostName_ << "'" << std::endl;
        std::cout << "   dbHostPort = '" << dbHostPort_ << "'" << std::endl;
      }
      std::cout << "   dbTagName = '" << dbTagName_ << "'" << std::endl;
#ifndef WITH_ECAL_COND_DB
      std::cout << std::endl;
      std::cout << "WARNING: DB access is NOT available" << std::endl;
      std::cout << std::endl;
#endif
    } else {
      std::cout << " Ecal Cond DB is OFF" << std::endl;
    }
  }

  // mergeRuns switch

  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);

  if ( verbose_ ) {
    if ( mergeRuns_ ) {
      std::cout << " mergeRuns switch is ON" << std::endl;
    } else {
      std::cout << " mergeRuns switch is OFF" << std::endl;
    }
  }

  // resetFile

  resetFile_ = ps.getUntrackedParameter<std::string>("resetFile", "");

  if ( verbose_ ) {
    if ( resetFile_.size() != 0 ) {
      std::cout << " resetFile is '" << resetFile_ << "'" << std::endl;
    }
  }

  // updateTime

  updateTime_ = ps.getUntrackedParameter<int>("updateTime", 0);

  if ( verbose_ ) {
    std::cout << " updateTime is " << updateTime_ << " minute(s)" << std::endl;
  }

  // dbUpdateTime

  dbUpdateTime_  = ps.getUntrackedParameter<int>("dbUpdateTime", 0);

  if ( verbose_ ) {
    std::cout << " dbUpdateTime is " << dbUpdateTime_ << " minute(s)" << std::endl;
  }

  // location

  location_ =  ps.getUntrackedParameter<std::string>("location", "H4");

  if ( verbose_ ) {
    std::cout << " location is '" << location_ << "'" << std::endl;
  }

  // cloneME switch

  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);

  if ( verbose_ ) {
    if ( cloneME_ ) {
      std::cout << " cloneME switch is ON" << std::endl;
    } else {
      std::cout << " cloneME switch is OFF" << std::endl;
    }
  }

  // debug switch

  debug_ = ps.getUntrackedParameter<bool>("debug", false);

  if ( verbose_ ) {
    if ( debug_ ) {
      std::cout << " debug switch is ON" << std::endl;
    } else {
      std::cout << " debug switch is OFF" << std::endl;
    }
  }

  // prescaleFactor

  prescaleFactor_ = ps.getUntrackedParameter<int>("prescaleFactor", 1);

  if ( verbose_ ) {
    std::cout << " prescaleFactor is " << prescaleFactor_ << std::endl;
  }

  // prefixME path

  prefixME_ = ps.getUntrackedParameter<std::string>("prefixME", "");

  if ( verbose_ ) {
    std::cout << " prefixME path is '" << prefixME_ << "'" << std::endl;
  }

  produceReports_ = ps.getUntrackedParameter<bool>("produceReports", true);

  if (produceReports_){
    std::cout << " producing reportSummaries" << std::endl;
  }

  // enableCleanup switch

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  if ( verbose_ ) {
    if ( enableCleanup_ ) {
      std::cout << " enableCleanup switch is ON" << std::endl;
    } else {
      std::cout << " enableCleanup switch is OFF" << std::endl;
    }
  }

  // vector of selected Super Modules (Defaults to all 18).

  superModules_.reserve(18);
  for ( unsigned int i = 1; i <= 18; i++ ) superModules_.push_back(i);

  superModules_ = ps.getUntrackedParameter<std::vector<int> >("superModules", superModules_);

  if ( verbose_ ) {
    std::cout << " Selected SMs:" << std::endl;
    for ( unsigned int i = 0; i < superModules_.size(); i++ ) {
      std::cout << " " << std::setw(2) << std::setfill('0') << superModules_[i];
    }
    std::cout << std::endl;
  }

  // vector of enabled Clients (defaults)

  enabledClients_.push_back("Integrity");
  enabledClients_.push_back("StatusFlags");
  enabledClients_.push_back("PedestalOnline");
  enabledClients_.push_back("Summary");

  enabledClients_ = ps.getUntrackedParameter<std::vector<std::string> >("enabledClients", enabledClients_);

  if ( verbose_ ) {
    std::cout << " Enabled Clients:" << std::endl;
    for ( unsigned int i = 0; i < enabledClients_.size(); i++ ) {
      std::cout << " " << enabledClients_[i];
    }
    std::cout << std::endl;
  }

  // set runTypes (use resize() on purpose!)

  runTypes_.resize(30);
  for ( unsigned int i = 0; i < runTypes_.size(); i++ ) runTypes_[i] =  "UNKNOWN";

  runTypes_[EcalDCCHeaderBlock::COSMIC]               = "COSMIC";
  runTypes_[EcalDCCHeaderBlock::BEAMH4]               = "BEAM";
  runTypes_[EcalDCCHeaderBlock::BEAMH2]               = "BEAM";
  runTypes_[EcalDCCHeaderBlock::MTCC]                 = "MTCC";
  runTypes_[EcalDCCHeaderBlock::LASER_STD]            = "LASER";
  runTypes_[EcalDCCHeaderBlock::LED_STD]              = "LED";
  runTypes_[EcalDCCHeaderBlock::TESTPULSE_MGPA]       = "TEST_PULSE";
  runTypes_[EcalDCCHeaderBlock::PEDESTAL_STD]         = "PEDESTAL";
  runTypes_[EcalDCCHeaderBlock::PEDESTAL_OFFSET_SCAN] = "PEDESTAL-OFFSET";

  runTypes_[EcalDCCHeaderBlock::COSMICS_GLOBAL]       = "COSMIC";
  runTypes_[EcalDCCHeaderBlock::PHYSICS_GLOBAL]       = "PHYSICS";
  runTypes_[EcalDCCHeaderBlock::HALO_GLOBAL]          = "HALO";
  runTypes_[EcalDCCHeaderBlock::COSMICS_LOCAL]        = "COSMIC";
  runTypes_[EcalDCCHeaderBlock::PHYSICS_LOCAL]        = "PHYSICS";
  runTypes_[EcalDCCHeaderBlock::HALO_LOCAL]           = "HALO";

  runTypes_[EcalDCCHeaderBlock::LASER_GAP]            = "LASER";
  runTypes_[EcalDCCHeaderBlock::LED_GAP]              = "LED";
  runTypes_[EcalDCCHeaderBlock::TESTPULSE_GAP]        = "TEST_PULSE";
  runTypes_[EcalDCCHeaderBlock::PEDESTAL_GAP]         = "PEDESTAL";

  runTypes_[EcalDCCHeaderBlock::CALIB_LOCAL]          = "CALIB";

  // clients' constructors

  clients_.reserve(12);
  clientsNames_.reserve(12);

  if ( find(enabledClients_.begin(), enabledClients_.end(), "Integrity" ) != enabledClients_.end() ) {

    clients_.push_back( new EEIntegrityClient(ps) );
    clientsNames_.push_back( "Integrity" );

    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMIC ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LASER_STD ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LED_STD ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PEDESTAL_STD ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PEDESTAL_OFFSET_SCAN ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::TESTPULSE_MGPA ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::BEAMH4 ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::BEAMH2 ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::MTCC ));

    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMICS_GLOBAL ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PHYSICS_GLOBAL ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMICS_LOCAL ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PHYSICS_LOCAL ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LASER_GAP ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LED_GAP ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::TESTPULSE_GAP ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PEDESTAL_GAP ));

  }

  if ( find(enabledClients_.begin(), enabledClients_.end(), "StatusFlags" ) != enabledClients_.end() ) {

    clients_.push_back( new EEStatusFlagsClient(ps) );
    clientsNames_.push_back( "StatusFlags" );

    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMIC ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LASER_STD ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LED_STD ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PEDESTAL_STD ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PEDESTAL_OFFSET_SCAN ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::TESTPULSE_MGPA ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::BEAMH4 ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::BEAMH2 ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::MTCC ));

    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMICS_GLOBAL ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PHYSICS_GLOBAL ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMICS_LOCAL ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PHYSICS_LOCAL ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LASER_GAP ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LED_GAP ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::TESTPULSE_GAP ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PEDESTAL_GAP ));

  }

  if ( find(enabledClients_.begin(), enabledClients_.end(), "Occupancy" ) != enabledClients_.end() ) {

    clients_.push_back( new EEOccupancyClient(ps) );
    clientsNames_.push_back( "Occupancy" );

    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMIC ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LASER_STD ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LED_STD ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PEDESTAL_STD ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PEDESTAL_OFFSET_SCAN ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::TESTPULSE_MGPA ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::BEAMH4 ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::BEAMH2 ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::MTCC ));

    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMICS_GLOBAL ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PHYSICS_GLOBAL ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMICS_LOCAL ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PHYSICS_LOCAL ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LASER_GAP ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LED_GAP ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::TESTPULSE_GAP ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PEDESTAL_GAP ));

  }

  if ( find(enabledClients_.begin(), enabledClients_.end(), "Cosmic" ) != enabledClients_.end() ) {

    clients_.push_back( new EECosmicClient(ps) );
    clientsNames_.push_back( "Cosmic" );

    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMIC ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LASER_STD ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LED_STD ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PEDESTAL_STD ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::TESTPULSE_MGPA ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::MTCC ));

    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMICS_GLOBAL ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PHYSICS_GLOBAL ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMICS_LOCAL ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PHYSICS_LOCAL ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LASER_GAP ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LED_GAP ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::TESTPULSE_GAP ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PEDESTAL_GAP ));

  }

  if ( find(enabledClients_.begin(), enabledClients_.end(), "Laser" ) != enabledClients_.end() ) {

    clients_.push_back( new EELaserClient(ps) );
    clientsNames_.push_back( "Laser" );

    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMIC ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LASER_STD ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LED_STD ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PEDESTAL_STD ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::TESTPULSE_MGPA ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::BEAMH4 ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::BEAMH2 ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::MTCC ));

    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMICS_GLOBAL ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PHYSICS_GLOBAL ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMICS_LOCAL ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PHYSICS_LOCAL ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LASER_GAP ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LED_GAP ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::TESTPULSE_GAP ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PEDESTAL_GAP ));

  }

  if ( find(enabledClients_.begin(), enabledClients_.end(), "Pedestal" ) != enabledClients_.end() ) {

    clients_.push_back( new EEPedestalClient(ps) );
    clientsNames_.push_back( "Pedestal" );

    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMIC ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LASER_STD ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LED_STD ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PEDESTAL_STD ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::TESTPULSE_MGPA ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::BEAMH4 ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::BEAMH2 ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::MTCC ));

    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMICS_GLOBAL ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PHYSICS_GLOBAL ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMICS_LOCAL ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PHYSICS_LOCAL ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LASER_GAP ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LED_GAP ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::TESTPULSE_GAP ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PEDESTAL_GAP ));

  }

  if ( find(enabledClients_.begin(), enabledClients_.end(), "PedestalOnline" ) != enabledClients_.end() ) {

    clients_.push_back( new EEPedestalOnlineClient(ps) );
    clientsNames_.push_back( "PedestalOnline" );

    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMIC ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LASER_STD ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LED_STD ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PEDESTAL_STD ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::TESTPULSE_MGPA ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::BEAMH4 ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::BEAMH2 ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::MTCC ));

    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMICS_GLOBAL ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PHYSICS_GLOBAL ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMICS_LOCAL ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PHYSICS_LOCAL ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LASER_GAP ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LED_GAP ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::TESTPULSE_GAP ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PEDESTAL_GAP ));

  }

  if ( find(enabledClients_.begin(), enabledClients_.end(), "TestPulse" ) != enabledClients_.end() ) {

    clients_.push_back( new EETestPulseClient(ps) );
    clientsNames_.push_back( "TestPulse" );

    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMIC ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LASER_STD ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LED_STD ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PEDESTAL_STD ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::TESTPULSE_MGPA ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::BEAMH4 ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::BEAMH2 ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::MTCC ));

    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMICS_GLOBAL ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PHYSICS_GLOBAL ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMICS_LOCAL ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PHYSICS_LOCAL ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LASER_GAP ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LED_GAP ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::TESTPULSE_GAP ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PEDESTAL_GAP ));

  }

  if ( find(enabledClients_.begin(), enabledClients_.end(), "TriggerTower" ) != enabledClients_.end() ) {

    clients_.push_back( new EETriggerTowerClient(ps) );
    clientsNames_.push_back( "TriggerTower" );

    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMIC ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LASER_STD ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LED_STD ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PEDESTAL_STD ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::TESTPULSE_MGPA ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::BEAMH4 ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::BEAMH2 ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::MTCC ));

    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMICS_GLOBAL ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PHYSICS_GLOBAL ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMICS_LOCAL ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PHYSICS_LOCAL ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LASER_GAP ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LED_GAP ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::TESTPULSE_GAP ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PEDESTAL_GAP ));

  }

  if ( find(enabledClients_.begin(), enabledClients_.end(), "Cluster" ) != enabledClients_.end() ) {

    clients_.push_back( new EEClusterClient(ps) );
    clientsNames_.push_back( "Cluster" );

    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMIC ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LASER_STD ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LED_STD ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PEDESTAL_STD ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::TESTPULSE_MGPA ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::BEAMH4 ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::BEAMH2 ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::MTCC ));

    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMICS_GLOBAL ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PHYSICS_GLOBAL ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMICS_LOCAL ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PHYSICS_LOCAL ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LASER_GAP ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LED_GAP ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::TESTPULSE_GAP ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PEDESTAL_GAP ));

  }

  if ( find(enabledClients_.begin(), enabledClients_.end(), "Timing" ) != enabledClients_.end() ) {

    clients_.push_back( new EETimingClient(ps) );
    clientsNames_.push_back( "Timing" );

    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMIC ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LASER_STD ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LED_STD ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PEDESTAL_STD ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::TESTPULSE_MGPA ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::BEAMH4 ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::BEAMH2 ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::MTCC ));

    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMICS_GLOBAL ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PHYSICS_GLOBAL ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMICS_LOCAL ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PHYSICS_LOCAL ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LASER_GAP ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LED_GAP ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::TESTPULSE_GAP ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PEDESTAL_GAP ));

  }

  if ( find(enabledClients_.begin(), enabledClients_.end(), "Led" ) != enabledClients_.end() ) {

    clients_.push_back( new EELedClient(ps) );
    clientsNames_.push_back( "Led" );

    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMIC ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LASER_STD ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LED_STD ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PEDESTAL_STD ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::TESTPULSE_MGPA ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::BEAMH4 ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::BEAMH2 ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::MTCC ));

    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PHYSICS_GLOBAL ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PHYSICS_LOCAL ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LASER_GAP ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LED_GAP ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::TESTPULSE_GAP ));
    clientsRuns_.insert(std::pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PEDESTAL_GAP ));

  }

  // define status bits

  clientsStatus_.insert(std::pair<std::string,int>( "Integrity",       0 ));
  clientsStatus_.insert(std::pair<std::string,int>( "Cosmic",          1 ));
  clientsStatus_.insert(std::pair<std::string,int>( "Laser",           2 ));
  clientsStatus_.insert(std::pair<std::string,int>( "Pedestal",        3 ));
  clientsStatus_.insert(std::pair<std::string,int>( "PedestalOnline",  4 ));
  clientsStatus_.insert(std::pair<std::string,int>( "TestPulse",       5 ));
  clientsStatus_.insert(std::pair<std::string,int>( "TriggerTower",    8 ));
  clientsStatus_.insert(std::pair<std::string,int>( "Cluster",         9 ));
  clientsStatus_.insert(std::pair<std::string,int>( "Timing",         10 ));
  clientsStatus_.insert(std::pair<std::string,int>( "Led",            11 ));
  clientsStatus_.insert(std::pair<std::string,int>( "StatusFlags",    12 ));
  clientsStatus_.insert(std::pair<std::string,int>( "Occupancy",      13 ));

  summaryClient_ = 0;

  if ( find(enabledClients_.begin(), enabledClients_.end(), "Summary" ) != enabledClients_.end() ) {

    summaryClient_ = new EESummaryClient(ps);

  }

  if ( summaryClient_ ) summaryClient_->setFriends(clients_);

  if ( verbose_ ) std::cout << std::endl;

}

EcalEndcapMonitorClient::~EcalEndcapMonitorClient() {

  if ( verbose_ ) std::cout << "Exit ..." << std::endl;

  for ( unsigned int i=0; i<clients_.size(); i++ ) {
    delete clients_[i];
  }

  if ( summaryClient_ ) delete summaryClient_;

}

void EcalEndcapMonitorClient::beginJob(void) {

  begin_run_ = false;
  end_run_   = false;

  forced_status_ = false;
  forced_update_ = false;

  h_ = 0;

  status_  = "unknown";

  run_     = -1;
  evt_     = -1;

  runType_ = -1;
  evtType_ = -1;

  last_run_ = -1;

  subrun_  = -1;

  if ( debug_ ) std::cout << "EcalEndcapMonitorClient: beginJob" << std::endl;

  ievt_ = 0;
  jevt_ = 0;

  current_time_ = time(NULL);
  last_time_update_ = current_time_;
  last_time_reset_ = current_time_;

  // get hold of back-end interface
  dqmStore_ = edm::Service<DQMStore>().operator->();

  if ( inputFile_.size() != 0 ) {
    if ( dqmStore_ ) {
      dqmStore_->open(inputFile_);
    }
  }

  for ( unsigned int i=0; i<clients_.size(); i++ ) {
    clients_[i]->beginJob();
  }

  if ( summaryClient_ ) summaryClient_->beginJob();

}

void EcalEndcapMonitorClient::beginRun(void) {

  begin_run_ = true;
  end_run_   = false;

  last_run_  = run_;

  if ( debug_ ) std::cout << "EcalEndcapMonitorClient: beginRun" << std::endl;

  jevt_ = 0;

  current_time_ = time(NULL);
  last_time_update_ = current_time_;
  last_time_reset_ = current_time_;

  this->setup();

  this->beginRunDb();

  for ( int i=0; i<int(clients_.size()); i++ ) {
    clients_[i]->cleanup();
    bool done = false;
    for ( std::multimap<EEClient*,int>::iterator j = clientsRuns_.lower_bound(clients_[i]); j != clientsRuns_.upper_bound(clients_[i]); j++ ) {
      if ( runType_ != -1 && runType_ == (*j).second && !done ) {
        done = true;
        clients_[i]->beginRun();
      }
    }
  }

  if ( summaryClient_ ) summaryClient_->beginRun();

}

void EcalEndcapMonitorClient::beginRun(const edm::Run& r, const edm::EventSetup& c) {

  Numbers::initGeometry(c, verbose_);

  if ( verbose_ ) std::cout << std::endl;

  Masks::initMasking(c, verbose_);

  if ( verbose_ ) {
    std::cout << std::endl;
    std::cout << "Standard beginRun() for run " << r.id().run() << std::endl;
    std::cout << std::endl;
  }

  // summary for DQM GUI

  if(produceReports_){

    MonitorElement* me;

    dqmStore_->setCurrentFolder( prefixME_ + "/EventInfo" );

    me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummary");
    if ( me ) {
      dqmStore_->removeElement(me->getName());
    }
    me = dqmStore_->bookFloat("reportSummary");
    me->Fill(-1.0);

    dqmStore_->setCurrentFolder( prefixME_ + "/EventInfo/reportSummaryContents" );

    for (int i = 0; i < 18; i++) {
      me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummaryContents/EcalEndcap_" + Numbers::sEE(i+1) );
      if ( me ) {
	dqmStore_->removeElement(me->getName());
      }
      me = dqmStore_->bookFloat("EcalEndcap_" + Numbers::sEE(i+1));
      me->Fill(-1.0);
    }

    dqmStore_->setCurrentFolder( prefixME_ + "/EventInfo" );

    me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummaryMap");
    if ( me ) {
      dqmStore_->removeElement(me->getName());
    }

    me = dqmStore_->book2D("reportSummaryMap", "EcalEndcap Report Summary Map", 40, 0., 200., 20, 0., 100);
    for ( int jx = 1; jx <= 40; jx++ ) {
      for ( int jy = 1; jy <= 20; jy++ ) {
	me->setBinContent( jx, jy, -1.0 );
      }
    }
    me->setAxisTitle("ix / ix+100", 1);
    me->setAxisTitle("iy", 2);
  }

  run_ = r.id().run();
  evt_ = 0;

  jevt_ = 0;

}

void EcalEndcapMonitorClient::endJob(void) {

  if ( ! end_run_ ) {

    if ( verbose_ ) {
      std::cout << std::endl;
      std::cout << "Checking last event at endJob() ... " << std::endl;
      std::cout << std::endl;
    }

    forced_update_ = true;

    this->analyze();

    if ( begin_run_ && ! end_run_ ) {

      if ( verbose_ ) {
        std::cout << std::endl;
        std::cout << "Forcing endRun() ... " << std::endl;
        std::cout << std::endl;
      }

      forced_status_ = true;

      this->analyze();
      this->endRun();

    }

  }

  if ( debug_ ) std::cout << "EcalEndcapMonitorClient: endJob, ievt = " << ievt_ << std::endl;

  this->cleanup();

  for ( unsigned int i=0; i<clients_.size(); i++ ) {
    clients_[i]->endJob();
  }

  if ( summaryClient_ ) summaryClient_->endJob();

}

void EcalEndcapMonitorClient::endRun(void) {

  begin_run_ = false;
  end_run_   = true;

  if ( debug_ ) std::cout << "EcalEndcapMonitorClient: endRun, jevt = " << jevt_ << std::endl;

  if ( subrun_ != -1 ) {

    this->writeDb();

    this->endRunDb();

  }

  if ( resetFile_.size() != 0 || dbUpdateTime_ > 0 ) {

    this->softReset(false);

    for ( int i=0; i<int(clients_.size()); i++ ) {
      bool done = false;
      for ( std::multimap<EEClient*,int>::iterator j = clientsRuns_.lower_bound(clients_[i]); j != clientsRuns_.upper_bound(clients_[i]); j++ ) {
        if ( runType_ != -1 && runType_ == (*j).second && !done ) {
          done = true;
          clients_[i]->analyze();
        }
      }
    }

    if ( summaryClient_ ) summaryClient_->analyze();

  }

  for ( int i=0; i<int(clients_.size()); i++ ) {
    bool done = false;
    for ( std::multimap<EEClient*,int>::iterator j = clientsRuns_.lower_bound(clients_[i]); j != clientsRuns_.upper_bound(clients_[i]); j++ ) {
      if ( runType_ != -1 && runType_ == (*j).second && !done ) {
        done = true;
        clients_[i]->endRun();
      }
    }
  }

  if ( summaryClient_ ) summaryClient_->endRun();

  this->cleanup();

  status_  = "unknown";

  run_     = -1;
  evt_     = -1;

  runType_ = -1;
  evtType_ = -1;

  subrun_ = -1;

}

void EcalEndcapMonitorClient::endRun(const edm::Run& r, const edm::EventSetup& c) {

  if ( verbose_ ) {
    std::cout << std::endl;
    std::cout << "Standard endRun() for run " << r.id().run() << std::endl;
    std::cout << std::endl;
  }


  this->analyze();

  if ( run_ != -1 && evt_ != -1 && runType_ != -1 ) {

    forced_update_ = true;

    this->analyze();

    if ( ! mergeRuns_ ) {

      if ( begin_run_ && ! end_run_ ) {

        forced_status_ = false;
        this->endRun();

      }

    }

  }

  // summary for DQM GUI

  if ( run_ != -1 && evt_ != -1 && runType_ == -1 )  {

    MonitorElement* me;

    me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummary");
    if ( me ) me->Fill(-1.0);

    for (int i = 0; i < 18; i++) {
      me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummaryContents/EcalEndcap_" + Numbers::sEE(i+1));
      if ( me ) me->Fill(-1.0);
    }

    me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummaryMap");
    for ( int jx = 1; jx <= 40; jx++ ) {
      for ( int jy = 1; jy <= 20; jy++ ) {
        if ( me ) me->setBinContent( jx, jy, -1.0 );
      }
    }

  }

}

void EcalEndcapMonitorClient::beginLuminosityBlock(const edm::LuminosityBlock& l, const edm::EventSetup& c) {

  if ( verbose_ ) {
    std::cout << std::endl;
    std::cout << "Standard beginLuminosityBlock() for luminosity block " << l.id().luminosityBlock() << " of run " << l.id().run() << std::endl;
    std::cout << std::endl;
  }

}

void EcalEndcapMonitorClient::endLuminosityBlock(const edm::LuminosityBlock& l, const edm::EventSetup& c) {

  current_time_ = time(NULL);

  if ( verbose_ ) {
    std::cout << std::endl;
    std::cout << "Standard endLuminosityBlock() for luminosity block " << l.id().luminosityBlock() << " of run " << l.id().run() << std::endl;
    std::cout << std::endl;
  }

  if(begin_run_ && !end_run_){
    unsigned iC(0);
    for(; iC < enabledClients_.size(); iC++){
      std::string& name(enabledClients_[iC]);

      if(name == "Cluster" || name == "Cosmic" || name == "Occupancy" || name == "StatusFlags" || name == "Trend") continue;

      std::string dir(prefixME_ + "/EE" + name + "Client");
      if(!dqmStore_->dirExists(dir) || !dqmStore_->containsAnyMonitorable(dir)){
        std::vector<std::string>::iterator itr(std::find(clientsNames_.begin(), clientsNames_.end(), name));
        if(itr == clientsNames_.end()) continue; // something seriously wrong, but ignore
        std::cout << "EE" << name << "Client is missing plots; issuing beginRun" << std::endl;

        break;
      }
    }
    if(iC != enabledClients_.size()){
      forced_status_ = false;
      endRun();
      beginRun();
      run_ = l.id().run();
      evt_ = 0;
    }
  }

  if ( updateTime_ > 0 ) {
    if ( (current_time_ - last_time_update_) < 60 * updateTime_ ) {
      return;
    }
    last_time_update_ = current_time_;
  }

  if ( run_ != -1 && evt_ != -1 && runType_ != -1 ) {

    forced_update_ = true;

    this->analyze();

  }
}

void EcalEndcapMonitorClient::reset(void) {

}

void EcalEndcapMonitorClient::setup(void) {

}

void EcalEndcapMonitorClient::cleanup(void) {

  if ( ! enableCleanup_ ) return;

  if ( cloneME_ ) {
    if ( h_ ) delete h_;
  }

  h_ = 0;

}

void EcalEndcapMonitorClient::beginRunDb(void) {

  subrun_ = 0;

#ifdef WITH_ECAL_COND_DB
  EcalCondDBInterface* econn;

  econn = 0;

  if ( dbName_.size() != 0 ) {
    try {
      if ( verbose_ ) std::cout << "Opening DB connection with TNS_ADMIN ..." << std::endl;
      econn = new EcalCondDBInterface(dbName_, dbUserName_, dbPassword_);
      if ( verbose_ ) std::cout << "done." << std::endl;
    } catch (std::runtime_error &e) {
      std::cerr << e.what() << std::endl;
      if ( dbHostName_.size() != 0 ) {
        try {
          if ( verbose_ ) std::cout << "Opening DB connection without TNS_ADMIN ..." << std::endl;
          econn = new EcalCondDBInterface(dbHostName_, dbName_, dbUserName_, dbPassword_, dbHostPort_);
          if ( verbose_ ) std::cout << "done." << std::endl;
        } catch (std::runtime_error &e) {
          std::cerr << e.what() << std::endl;
        }
      }
    }
  }

  // create the objects necessary to identify a dataset

  LocationDef locdef;

  locdef.setLocation(location_);

  RunTypeDef rundef;

  rundef.setRunType( this->getRunType() );

  RunTag runtag;

  runtag.setLocationDef(locdef);
  runtag.setRunTypeDef(rundef);

  runtag.setGeneralTag( this->getRunType() );

  // fetch the RunIOV from the DB

  bool foundRunIOV = false;

  if ( econn ) {
    try {
      if ( verbose_ ) std::cout << "Fetching RunIOV ..." << std::endl;
//      runiov_ = econn->fetchRunIOV(&runtag, run_);
      runiov_ = econn->fetchRunIOV(location_, run_);
      if ( verbose_ ) std::cout << "done." << std::endl;
      foundRunIOV = true;
    } catch (std::runtime_error &e) {
      std::cerr << e.what() << std::endl;
      foundRunIOV = false;
    }
  }

  // begin - setup the RunIOV (on behalf of the DAQ)

  if ( ! foundRunIOV ) {

    Tm startRun;

    startRun.setToCurrentGMTime();

    runiov_.setRunNumber(run_);
    runiov_.setRunStart(startRun);
    runiov_.setRunTag(runtag);

    if ( econn ) {
      try {
        if ( verbose_ ) std::cout << "Inserting RunIOV ..." << std::endl;
        econn->insertRunIOV(&runiov_);
//        runiov_ = econn->fetchRunIOV(&runtag, run_);
        runiov_ = econn->fetchRunIOV(location_, run_);
        if ( verbose_ ) std::cout << "done." << std::endl;
      } catch (std::runtime_error &e) {
        std::cerr << e.what() << std::endl;
        try {
          if ( verbose_ ) std::cout << "Fetching RunIOV (again) ..." << std::endl;
//          runiov_ = econn->fetchRunIOV(&runtag, run_);
          runiov_ = econn->fetchRunIOV(location_, run_);
          if ( verbose_ ) std::cout << "done." << std::endl;
          foundRunIOV = true;
        } catch (std::runtime_error &e) {
          std::cerr << e.what() << std::endl;
          foundRunIOV = false;
        }
      }
    }

  }

  // end - setup the RunIOV (on behalf of the DAQ)

  if ( verbose_ ) {
    std::cout << std::endl;
    std::cout << "=============RunIOV:" << std::endl;
    std::cout << "Run Number:         " << runiov_.getRunNumber() << std::endl;
    std::cout << "Run Start:          " << runiov_.getRunStart().str() << std::endl;
    std::cout << "Run End:            " << runiov_.getRunEnd().str() << std::endl;
    std::cout << "====================" << std::endl;
    std::cout << std::endl;
    std::cout << "=============RunTag:" << std::endl;
    std::cout << "GeneralTag:         " << runiov_.getRunTag().getGeneralTag() << std::endl;
    std::cout << "Location:           " << runiov_.getRunTag().getLocationDef().getLocation() << std::endl;
    std::cout << "Run Type:           " << runiov_.getRunTag().getRunTypeDef().getRunType() << std::endl;
    std::cout << "====================" << std::endl;
    std::cout << std::endl;
  }

  std::string rt = runiov_.getRunTag().getRunTypeDef().getRunType();
  if ( strcmp(rt.c_str(), "UNKNOWN") == 0 ) {
    runType_ = -1;
  } else {
    for ( unsigned int i = 0; i < runTypes_.size(); i++ ) {
      if ( strcmp(rt.c_str(), runTypes_[i].c_str()) == 0 ) {
        if ( runType_ != int(i) ) {
          if ( verbose_ ) {
            std::cout << std::endl;
            std::cout << "Fixing Run Type to: " << runTypes_[i] << std::endl;
            std::cout << std::endl;
          }
          runType_ = i;
        }
        break;
      }
    }
  }

  if ( verbose_ ) std::cout << std::endl;

  if ( econn ) {
    try {
      if ( verbose_ ) std::cout << "Closing DB connection ..." << std::endl;
      delete econn;
      econn = 0;
      if ( verbose_ ) std::cout << "done." << std::endl;
    } catch (std::runtime_error &e) {
      std::cerr << e.what() << std::endl;
    }
  }
#endif

  if ( verbose_ ) std::cout << std::endl;

}

void EcalEndcapMonitorClient::writeDb(void) {

  subrun_++;

#ifdef WITH_ECAL_COND_DB
  EcalCondDBInterface* econn;

  econn = 0;

  if ( dbName_.size() != 0 ) {
    try {
      if ( verbose_ ) std::cout << "Opening DB connection with TNS_ADMIN ..." << std::endl;
      econn = new EcalCondDBInterface(dbName_, dbUserName_, dbPassword_);
      if ( verbose_ ) std::cout << "done." << std::endl;
    } catch (std::runtime_error &e) {
      std::cerr << e.what() << std::endl;
      if ( dbHostName_.size() != 0 ) {
        try {
          if ( verbose_ ) std::cout << "Opening DB connection without TNS_ADMIN ..." << std::endl;
          econn = new EcalCondDBInterface(dbHostName_, dbName_, dbUserName_, dbPassword_, dbHostPort_);
          if ( verbose_ ) std::cout << "done." << std::endl;
        } catch (std::runtime_error &e) {
          std::cerr << e.what() << std::endl;
        }
      }
    }
  }

  MonVersionDef monverdef;

  monverdef.setMonitoringVersion("test01");

  MonRunTag montag;

  montag.setMonVersionDef(monverdef);
  montag.setGeneralTag(dbTagName_);

  Tm startSubRun;

  startSubRun.setToCurrentGMTime();

  // fetch the MonIOV from the DB

  bool foundMonIOV = false;

  if ( econn ) {
    try {
      if ( verbose_ ) std::cout << "Fetching MonIOV ..." << std::endl;
      RunTag runtag = runiov_.getRunTag();
      moniov_ = econn->fetchMonRunIOV(&runtag, &montag, run_, subrun_);
      if ( verbose_ ) std::cout << "done." << std::endl;
      foundMonIOV = true;
    } catch (std::runtime_error &e) {
      std::cerr << e.what() << std::endl;
      foundMonIOV = false;
    }
  }

  // begin - setup the MonIOV

  if ( !foundMonIOV ) {

    moniov_.setRunIOV(runiov_);
    moniov_.setSubRunNumber(subrun_);

    if ( subrun_ > 1 ) {
      moniov_.setSubRunStart(startSubRun);
    } else {
      moniov_.setSubRunStart(runiov_.getRunStart());
    }

    moniov_.setMonRunTag(montag);

    if ( econn ) {
      try {
        if ( verbose_ ) std::cout << "Inserting MonIOV ..." << std::endl;
        econn->insertMonRunIOV(&moniov_);
        RunTag runtag = runiov_.getRunTag();
        moniov_ = econn->fetchMonRunIOV(&runtag, &montag, run_, subrun_);
        if ( verbose_ ) std::cout << "done." << std::endl;
      } catch (std::runtime_error &e) {
        std::cerr << e.what() << std::endl;
        try {
          if ( verbose_ ) std::cout << "Fetching MonIOV (again) ..." << std::endl;
          RunTag runtag = runiov_.getRunTag();
          moniov_ = econn->fetchMonRunIOV(&runtag, &montag, run_, subrun_);
          if ( verbose_ ) std::cout << "done." << std::endl;
          foundMonIOV = true;
        } catch (std::runtime_error &e) {
          std::cerr << e.what() << std::endl;
          foundMonIOV = false;
        }
      }
    }

  }

  // end - setup the MonIOV

  if ( verbose_ ) {
    std::cout << std::endl;
    std::cout << "==========MonRunIOV:" << std::endl;
    std::cout << "SubRun Number:      " << moniov_.getSubRunNumber() << std::endl;
    std::cout << "SubRun Start:       " << moniov_.getSubRunStart().str() << std::endl;
    std::cout << "SubRun End:         " << moniov_.getSubRunEnd().str() << std::endl;
    std::cout << "====================" << std::endl;
    std::cout << std::endl;
    std::cout << "==========MonRunTag:" << std::endl;
    std::cout << "GeneralTag:         " << moniov_.getMonRunTag().getGeneralTag() << std::endl;
    std::cout << "Monitoring Ver:     " << moniov_.getMonRunTag().getMonVersionDef().getMonitoringVersion() << std::endl;
    std::cout << "====================" << std::endl;
    std::cout << std::endl;
  }

  int taskl = 0x0;
  int tasko = 0x0;

  for ( int i=0; i<int(clients_.size()); i++ ) {
    bool done = false;
    for ( std::multimap<EEClient*,int>::iterator j = clientsRuns_.lower_bound(clients_[i]); j != clientsRuns_.upper_bound(clients_[i]); j++ ) {
      if ( h_ && runType_ != -1 && runType_ == (*j).second && !done ) {
        if ( strcmp(clientsNames_[i].c_str(), "Cosmic") == 0 && runType_ != EcalDCCHeaderBlock::COSMIC && runType_ != EcalDCCHeaderBlock::COSMICS_LOCAL && runType_ != EcalDCCHeaderBlock::COSMICS_GLOBAL && runType_ != EcalDCCHeaderBlock::PHYSICS_GLOBAL && runType_ != EcalDCCHeaderBlock::PHYSICS_LOCAL && h_->GetBinContent(2+EcalDCCHeaderBlock::COSMIC) == 0 && h_->GetBinContent(2+EcalDCCHeaderBlock::COSMICS_LOCAL) == 0 && h_->GetBinContent(2+EcalDCCHeaderBlock::COSMICS_GLOBAL) == 0 && h_->GetBinContent(2+EcalDCCHeaderBlock::PHYSICS_GLOBAL) == 0 && h_->GetBinContent(2+EcalDCCHeaderBlock::PHYSICS_LOCAL) == 0 ) continue;
        if ( strcmp(clientsNames_[i].c_str(), "Laser") == 0 && runType_ != EcalDCCHeaderBlock::LASER_STD && runType_ != EcalDCCHeaderBlock::LASER_GAP && h_->GetBinContent(2+EcalDCCHeaderBlock::LASER_STD) == 0 && h_->GetBinContent(2+EcalDCCHeaderBlock::LASER_GAP) == 0 ) continue;
        if ( strcmp(clientsNames_[i].c_str(), "Led") == 0 && runType_ != EcalDCCHeaderBlock::LED_STD && runType_ != EcalDCCHeaderBlock::LED_GAP && h_->GetBinContent(2+EcalDCCHeaderBlock::LED_STD) == 0 && h_->GetBinContent(2+EcalDCCHeaderBlock::LED_GAP) == 0 ) continue;
        if ( strcmp(clientsNames_[i].c_str(), "Pedestal") == 0 && runType_ != EcalDCCHeaderBlock::PEDESTAL_STD && runType_ != EcalDCCHeaderBlock::PEDESTAL_GAP && h_->GetBinContent(2+EcalDCCHeaderBlock::PEDESTAL_STD) == 0 && h_->GetBinContent(2+EcalDCCHeaderBlock::PEDESTAL_GAP) == 0 ) continue;
        if ( strcmp(clientsNames_[i].c_str(), "TestPulse") == 0 && runType_ != EcalDCCHeaderBlock::TESTPULSE_MGPA && runType_ != EcalDCCHeaderBlock::TESTPULSE_GAP && h_->GetBinContent(2+EcalDCCHeaderBlock::TESTPULSE_MGPA) == 0 && h_->GetBinContent(2+EcalDCCHeaderBlock::TESTPULSE_GAP) == 0 ) continue;
        done = true;
        if ( verbose_ ) {
          if ( econn ) {
            std::cout << " Writing " << clientsNames_[i] << " results to DB " << std::endl;
            std::cout << std::endl;
          }
        }
        bool status;
        if ( clients_[i]->writeDb(econn, &runiov_, &moniov_, status) ) {
          taskl |= 0x1 << clientsStatus_[clientsNames_[i]];
          if ( status ) {
            tasko |= 0x1 << clientsStatus_[clientsNames_[i]];
          }
        } else {
          tasko |= 0x1 << clientsStatus_[clientsNames_[i]];
        }
      }
    }
    if ( ((taskl >> clientsStatus_[clientsNames_[i]]) & 0x1) ) {
      if ( verbose_ ) {
        std::cout << " Task output for " << clientsNames_[i] << " = "
             << ((tasko >> clientsStatus_[clientsNames_[i]]) & 0x1) << std::endl;
        std::cout << std::endl;
      }
    }
  }

  bool status;
  if ( summaryClient_ ) summaryClient_->writeDb(econn, &runiov_, &moniov_, status);

  EcalLogicID ecid;
  MonRunDat md;
  std::map<EcalLogicID, MonRunDat> dataset;

  MonRunOutcomeDef monRunOutcomeDef;

  monRunOutcomeDef.setShortDesc("success");

  float nevt = -1.;

  if ( h_ ) nevt = h_->GetSumOfWeights();

  md.setNumEvents(int(nevt));
  md.setMonRunOutcomeDef(monRunOutcomeDef);

//  string fileName = "";
//  md.setRootfileName(fileName);

  md.setTaskList(taskl);
  md.setTaskOutcome(tasko);

  if ( econn ) {
    try {
      ecid = LogicID::getEcalLogicID("EE");
      dataset[ecid] = md;
    } catch (std::runtime_error &e) {
      std::cerr << e.what() << std::endl;
    }
  }

  if ( econn ) {
    try {
      if ( verbose_ ) std::cout << "Inserting MonRunDat ..." << std::endl;
      econn->insertDataSet(&dataset, &moniov_);
      if ( verbose_ ) std::cout << "done." << std::endl;
    } catch (std::runtime_error &e) {
      std::cerr << e.what() << std::endl;
    }
  }

  if ( econn ) {
    try {
      if ( verbose_ ) std::cout << "Closing DB connection ..." << std::endl;
      delete econn;
      econn = 0;
      if ( verbose_ ) std::cout << "done." << std::endl;
    } catch (std::runtime_error &e) {
      std::cerr << e.what() << std::endl;
    }
  }
#endif

  if ( verbose_ ) std::cout << std::endl;

}

void EcalEndcapMonitorClient::endRunDb(void) {

#ifdef WITH_ECAL_COND_DB
  EcalCondDBInterface* econn;

  econn = 0;

  if ( dbName_.size() != 0 ) {
    try {
      if ( verbose_ ) std::cout << "Opening DB connection with TNS_ADMIN ..." << std::endl;
      econn = new EcalCondDBInterface(dbName_, dbUserName_, dbPassword_);
      if ( verbose_ ) std::cout << "done." << std::endl;
    } catch (std::runtime_error &e) {
      std::cerr << e.what() << std::endl;
      if ( dbHostName_.size() != 0 ) {
        try {
          if ( verbose_ ) std::cout << "Opening DB connection without TNS_ADMIN ..." << std::endl;
          econn = new EcalCondDBInterface(dbHostName_, dbName_, dbUserName_, dbPassword_, dbHostPort_);
          if ( verbose_ ) std::cout << "done." << std::endl;
        } catch (std::runtime_error &e) {
          std::cerr << e.what() << std::endl;
        }
      }
    }
  }

  EcalLogicID ecid;
  RunDat rd;
  std::map<EcalLogicID, RunDat> dataset;

  float nevt = -1.;

  if ( h_ ) nevt = h_->GetSumOfWeights();

  rd.setNumEvents(int(nevt));

  // fetch the RunDat from the DB

  bool foundRunDat = false;

  if ( econn ) {
    try {
      if ( verbose_ ) std::cout << "Fetching RunDat ..." << std::endl;
      econn->fetchDataSet(&dataset, &runiov_);
      if ( verbose_ ) std::cout << "done." << std::endl;
      foundRunDat = true;
    } catch (std::runtime_error &e) {
      std::cerr << e.what() << std::endl;
      foundRunDat = false;
    }
  }

  // begin - setup the RunDat (on behalf of the DAQ)

  if ( ! foundRunDat ) {

    if ( econn ) {
      try {
        ecid = LogicID::getEcalLogicID("EE");
        dataset[ecid] = rd;
      } catch (std::runtime_error &e) {
        std::cerr << e.what() << std::endl;
      }
    }

    if ( econn ) {
      try {
        if ( verbose_ ) std::cout << "Inserting RunDat ..." << std::endl;
        econn->insertDataSet(&dataset, &runiov_);
        if ( verbose_ ) std::cout << "done." << std::endl;
      } catch (std::runtime_error &e) {
        std::cerr << e.what() << std::endl;
      }
    }

  }

  // end - setup the RunDat (on behalf of the DAQ)

  if ( econn ) {
    try {
      if ( verbose_ ) std::cout << "Closing DB connection ..." << std::endl;
      delete econn;
      econn = 0;
      if ( verbose_ ) std::cout << "done." << std::endl;
    } catch (std::runtime_error &e) {
      std::cerr << e.what() << std::endl;
    }
  }
#endif

}

void EcalEndcapMonitorClient::analyze(void) {

  current_time_ = time(NULL);

  ievt_++;
  jevt_++;

  if ( debug_ ) std::cout << "EcalEndcapMonitorClient: ievt/jevt = " << ievt_ << "/" << jevt_ << std::endl;

  MonitorElement* me;
  std::string s;

  me = dqmStore_->get(prefixME_ + "/EcalInfo/STATUS");
  if ( me ) {
    status_ = "unknown";
    s = me->valueString();
    if ( strcmp(s.c_str(), "i=0") == 0 ) status_ = "begin-of-run";
    if ( strcmp(s.c_str(), "i=1") == 0 ) status_ = "running";
    if ( strcmp(s.c_str(), "i=2") == 0 ) status_ = "end-of-run";
    if ( debug_ ) std::cout << "Found '" << prefixME_ << "/EcalInfo/STATUS'" << std::endl;
  }

  if ( inputFile_.size() != 0 ) {
    if ( ievt_ == 1 ) {
      if ( verbose_ ) {
        std::cout << std::endl;
        std::cout << " Reading DQM from file, forcing 'begin-of-run'" << std::endl;
        std::cout << std::endl;
      }
      status_ = "begin-of-run";
    }
  }

  int ecal_run = -1;
  me = dqmStore_->get(prefixME_ + "/EcalInfo/RUN");
  if ( me ) {
    s = me->valueString();
    sscanf(s.c_str(), "i=%d", &ecal_run);
    if ( debug_ ) std::cout << "Found '" << prefixME_ << "/EcalInfo/RUN'" << std::endl;
  }

  int ecal_evt = -1;
  me = dqmStore_->get(prefixME_ + "/EcalInfo/EVT");
  if ( me ) {
    s = me->valueString();
    sscanf(s.c_str(), "i=%d", &ecal_evt);
    if ( debug_ ) std::cout << "Found '" << prefixME_ << "/EcalInfo/EVT'" << std::endl;
  }

  me = dqmStore_->get(prefixME_ + "/EcalInfo/EVTTYPE");
  h_ = UtilsClient::getHisto<TH1F*>( me, cloneME_, h_ );

  me = dqmStore_->get(prefixME_ + "/EcalInfo/RUNTYPE");
  if ( me ) {
    s = me->valueString();
    sscanf(s.c_str(), "i=%d", &evtType_);
    if ( runType_ == -1 ) runType_ = evtType_;
    if ( debug_ ) std::cout << "Found '" << prefixME_ << "/EcalInfo/RUNTYPE' " << s << std::endl;
  }

  // if the run number from the Event is less than zero,
  // use the run number from the ECAL DCC header
  if ( run_ <= 0 ) run_ = ecal_run;

  if ( run_ != -1 && evt_ != -1 && runType_ != -1 ) {
    if ( ! mergeRuns_ && run_ != last_run_ ) forced_update_ = true;
  }

  bool update = ( forced_update_                    ) ||
                ( prescaleFactor_ != 1              ) ||
                ( jevt_ <   10                      ) ||
                ( jevt_ <  100 && jevt_ %   10 == 0 ) ||
                ( jevt_ < 1000 && jevt_ %  100 == 0 ) ||
                (                 jevt_ % 1000 == 0 );

  if ( update || strcmp(status_.c_str(), "begin-of-run") == 0 || strcmp(status_.c_str(), "end-of-run") == 0 ) {

    if ( verbose_ ) {
      std::cout << " RUN status = \"" << status_ << "\"" << std::endl;
      std::cout << "   CMS run/event number = " << run_ << "/" << evt_ << std::endl;
      std::cout << "   EE run/event number = " << ecal_run << "/" << ecal_evt << std::endl;
      std::cout << "   EE location = " << location_ << std::endl;
      std::cout << "   EE run/event type = " << this->getRunType() << "/" << ( evtType_ == -1 ? "UNKNOWN" : runTypes_[evtType_] ) << std::flush;

      if ( h_ ) {
        if ( h_->GetSumOfWeights() != 0 ) {
          std::cout << " ( " << std::flush;
          for ( unsigned int i = 0; i < runTypes_.size(); i++ ) {
            if ( strcmp(runTypes_[i].c_str(), "UNKNOWN") != 0 && h_->GetBinContent(2+i) != 0 ) {
              std::string s = runTypes_[i];
              transform( s.begin(), s.end(), s.begin(), (int(*)(int))tolower );
              std::cout << s << " ";
            }
          }
          std::cout << ")" << std::flush;
        }
      }
      std::cout << std::endl;
    }

  }

  if ( strcmp(status_.c_str(), "begin-of-run") == 0 ) {

    if ( run_ != -1 && evt_ != -1 && runType_ != -1 ) {

      if ( ! begin_run_ ) {

        forced_status_ = false;
        this->beginRun();

      }

    }

  }

  if ( strcmp(status_.c_str(), "begin-of-run") == 0 || strcmp(status_.c_str(), "running") == 0 || strcmp(status_.c_str(), "end-of-run") == 0 ) {

    if ( begin_run_ && ! end_run_ ) {

      bool update = ( forced_update_                      ) ||
                    ( prescaleFactor_ != 1                ) ||
                    ( jevt_ <     3                       ) ||
                    ( jevt_ <  1000 && jevt_ %   100 == 0 ) ||
                    ( jevt_ < 10000 && jevt_ %  1000 == 0 ) ||
                    (                  jevt_ % 10000 == 0 );

      if ( update || strcmp(status_.c_str(), "begin-of-run") == 0 || strcmp(status_.c_str(), "end-of-run") == 0 ) {

        for ( int i=0; i<int(clients_.size()); i++ ) {
          bool done = false;
          for ( std::multimap<EEClient*,int>::iterator j = clientsRuns_.lower_bound(clients_[i]); j != clientsRuns_.upper_bound(clients_[i]); j++ ) {
            if ( runType_ != -1 && runType_ == (*j).second && !done ) {
              done = true;
              clients_[i]->analyze();
            }
          }
        }

        if ( summaryClient_ ) summaryClient_->analyze();

      }

      forced_update_ = false;

      bool reset = false;

      if ( resetFile_.size() != 0 ) {
        if ( access(resetFile_.c_str(), W_OK) == 0 ) {
          if ( unlink(resetFile_.c_str()) == 0 ) {
            reset |= true;
          }
        }
      }

      if ( dbUpdateTime_ > 0 ) {
        reset |= (current_time_ - last_time_reset_) > 60 * dbUpdateTime_;
      }

      if ( reset ) {
        if ( runType_ == EcalDCCHeaderBlock::COSMIC ||
             runType_ == EcalDCCHeaderBlock::COSMICS_GLOBAL ||
             runType_ == EcalDCCHeaderBlock::PHYSICS_GLOBAL ||
             runType_ == EcalDCCHeaderBlock::COSMICS_LOCAL ||
             runType_ == EcalDCCHeaderBlock::PHYSICS_LOCAL ||
             runType_ == EcalDCCHeaderBlock::BEAMH2 ||
             runType_ == EcalDCCHeaderBlock::BEAMH4 ) this->writeDb();
        this->softReset(true);
        last_time_reset_ = current_time_;
      }

    }

  }

  if ( strcmp(status_.c_str(), "end-of-run") == 0 ) {

    if ( run_ != -1 && evt_ != -1 && runType_ != -1 ) {

      if ( begin_run_ && ! end_run_ ) {

        forced_status_ = false;
        this->endRun();

      }

    }

  }

  // BEGIN: run-time fixes for missing state transitions

  // run number transition

  if ( strcmp(status_.c_str(), "running") == 0 ) {

    if ( run_ != -1 && evt_ != -1 && runType_ != -1 ) {

      if ( ! mergeRuns_ ) {

        int new_run_ = run_;
        int old_run_ = last_run_;

        if ( new_run_ != old_run_ ) {

          if ( begin_run_ && ! end_run_ ) {

            if ( verbose_ ) {
              std::cout << std::endl;
              std::cout << " Old run has finished, issuing endRun() ... " << std::endl;
              std::cout << std::endl;
            }

            // end old_run_
            run_ = old_run_;

            forced_status_ = false;
            this->endRun();

          }

          if ( ! begin_run_ ) {

            if ( verbose_ ) {
              std::cout << std::endl;
              std::cout << " New run has started, issuing beginRun() ... " << std::endl;
              std::cout << std::endl;
            }

            // start new_run_
            run_ = new_run_;

            forced_status_ = false;
            this->beginRun();

          }

        }

      }

    }

  }

  // 'running' state without a previous 'begin-of-run' state

  if ( strcmp(status_.c_str(), "running") == 0 ) {

    if ( run_ != -1 && evt_ != -1 && runType_ != -1 ) {

      if ( ! forced_status_ ) {

        if ( ! begin_run_ ) {

          if ( verbose_ ) {
            std::cout << std::endl;
            std::cout << "Forcing beginRun() ... NOW !" << std::endl;
            std::cout << std::endl;
          }

          forced_status_ = true;
          this->beginRun();

        }

      }

    }

  }

  // 'end-of-run' state without a previous 'begin-of-run' or 'running' state

  if ( strcmp(status_.c_str(), "end-of-run") == 0 ) {

    if ( run_ != -1 && evt_ != -1 && runType_ != -1 ) {

      if ( ! forced_status_ ) {

        if ( ! begin_run_ ) {

          if ( verbose_ ) {
            std::cout << std::endl;
            std::cout << "Forcing beginRun() ... NOW !" << std::endl;
            std::cout << std::endl;
          }

          forced_status_ = true;
          this->beginRun();

        }

      }

    }

  }

  // END: run-time fixes for missing state transitions

}

void EcalEndcapMonitorClient::analyze(const edm::Event& e, const edm::EventSetup& c) {

  run_ = e.id().run();
  evt_ = e.id().event();

  if ( prescaleFactor_ > 0 ) {
    if ( jevt_ % prescaleFactor_ == 0 ){
      this->analyze();
    }
  }

}

void EcalEndcapMonitorClient::softReset(bool flag) {

  std::vector<MonitorElement*> mes = dqmStore_->getAllContents(prefixME_);
  std::vector<MonitorElement*>::const_iterator meitr;
  for ( meitr=mes.begin(); meitr!=mes.end(); meitr++ ) {
    if ( !strncmp((*meitr)->getName().c_str(), "EE", 2)
         && strncmp((*meitr)->getName().c_str(), "EETrend", 7)
         && strncmp((*meitr)->getName().c_str(), "by lumi", 7) ) {
      if ( flag ) {
        dqmStore_->softReset(*meitr);
      } else {
        dqmStore_->disableSoftReset(*meitr);
      }
    }
  }

  MonitorElement* me = dqmStore_->get(prefixME_ + "/EcalInfo/EVTTYPE");
  if ( me ) {
    if ( flag ) {
      dqmStore_->softReset(me);
    } else {
      dqmStore_->disableSoftReset(me);
    }
  }

}


/*
 * \file EcalEndcapMonitorClient.cc
 *
 * $Date: 2009/07/02 12:23:03 $
 * $Revision: 1.207 $
 * \author G. Della Ricca
 * \author F. Cossutti
 *
*/

#include <memory>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>

#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQMServices/Core/interface/DQMOldReceiver.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/EcalRawData/interface/EcalDCCHeaderBlock.h"

#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"
#include "OnlineDB/EcalCondDB/interface/RunTag.h"
#include "OnlineDB/EcalCondDB/interface/RunDat.h"
#include "OnlineDB/EcalCondDB/interface/MonRunDat.h"

#include "DQM/EcalCommon/interface/EcalErrorMask.h"
#include <DQM/EcalCommon/interface/UtilsClient.h>
#include <DQM/EcalCommon/interface/Numbers.h>
#include <DQM/EcalCommon/interface/LogicID.h>

#include <DQM/EcalEndcapMonitorClient/interface/EcalEndcapMonitorClient.h>

#include <DQM/EcalEndcapMonitorClient/interface/EEIntegrityClient.h>
#include <DQM/EcalEndcapMonitorClient/interface/EEStatusFlagsClient.h>
#include <DQM/EcalEndcapMonitorClient/interface/EEOccupancyClient.h>
#include <DQM/EcalEndcapMonitorClient/interface/EECosmicClient.h>
#include <DQM/EcalEndcapMonitorClient/interface/EELaserClient.h>
#include <DQM/EcalEndcapMonitorClient/interface/EEPedestalClient.h>
#include <DQM/EcalEndcapMonitorClient/interface/EEPedestalOnlineClient.h>
#include <DQM/EcalEndcapMonitorClient/interface/EETestPulseClient.h>
#include <DQM/EcalEndcapMonitorClient/interface/EEBeamCaloClient.h>
#include <DQM/EcalEndcapMonitorClient/interface/EEBeamHodoClient.h>
#include <DQM/EcalEndcapMonitorClient/interface/EETriggerTowerClient.h>
#include <DQM/EcalEndcapMonitorClient/interface/EEClusterClient.h>
#include <DQM/EcalEndcapMonitorClient/interface/EETimingClient.h>
#include <DQM/EcalEndcapMonitorClient/interface/EELedClient.h>

using namespace cms;
using namespace edm;
using namespace std;

EcalEndcapMonitorClient::EcalEndcapMonitorClient(const ParameterSet& ps) {

  // verbose switch

  verbose_ = ps.getUntrackedParameter<bool>("verbose",  true);

  if ( verbose_ ) {
    cout << endl;
    cout << " *** Ecal Endcap Generic Monitor Client ***" << endl;
    cout << endl;
  }

  // DQM ROOT input file

  inputFile_ = ps.getUntrackedParameter<string>("inputFile", "");

  if ( verbose_ ) {
    if ( inputFile_.size() != 0 ) {
      cout << " Reading DQM data from inputFile = '" << inputFile_ << "'" << endl;
    }
  }

  // Ecal Cond DB

  dbName_ = ps.getUntrackedParameter<string>("dbName", "");
  dbHostName_ = ps.getUntrackedParameter<string>("dbHostName", "");
  dbHostPort_ = ps.getUntrackedParameter<int>("dbHostPort", 1521);
  dbUserName_ = ps.getUntrackedParameter<string>("dbUserName", "");
  dbPassword_ = ps.getUntrackedParameter<string>("dbPassword", "");

  dbTagName_ = ps.getUntrackedParameter<string>("dbTagName", "CMSSW");

  if ( verbose_ ) {
    if ( dbName_.size() != 0 ) {
      cout << " Using Ecal Cond DB: " << endl;
      cout << "   dbName = '" << dbName_ << "'" << endl;
      cout << "   dbUserName = '" << dbUserName_ << "'" << endl;
      if ( dbHostName_.size() != 0 ) {
        cout << "   dbHostName = '" << dbHostName_ << "'" << endl;
        cout << "   dbHostPort = '" << dbHostPort_ << "'" << endl;
      }
      cout << "   dbTagName = '" << dbTagName_ << "'" << endl;
    } else {
      cout << " Ecal Cond DB is OFF" << endl;
    }
  }

  // Mask file

  maskFile_ = ps.getUntrackedParameter<string>("maskFile", "");

  if ( verbose_ ) {
    if ( maskFile_.size() != 0 ) {
      cout << " Using maskFile = '" << maskFile_ << "'" << endl;
    }
  }

  // mergeRuns switch

  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);

  if ( verbose_ ) {
    if ( mergeRuns_ ) {
      cout << " mergeRuns switch is ON" << endl;
    } else {
      cout << " mergeRuns switch is OFF" << endl;
    }
  }

  // updateTime

  updateTime_ = ps.getUntrackedParameter<int>("updateTime", 0);

  if ( verbose_ ) {
    cout << " updateTime is " << updateTime_ << " minute(s)" << endl;
  }

  // dbUpdateTime

  dbUpdateTime_  = ps.getUntrackedParameter<int>("dbUpdateTime", 0);

  if ( verbose_ ) {
    cout << " dbUpdateTime is " << dbUpdateTime_ << " minute(s)" << endl;
  }

  // location

  location_ =  ps.getUntrackedParameter<string>("location", "H4");

  if ( verbose_ ) {
    cout << " location is '" << location_ << "'" << endl;
  }

  // cloneME switch

  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);

  if ( verbose_ ) {
    if ( cloneME_ ) {
      cout << " cloneME switch is ON" << endl;
    } else {
      cout << " cloneME switch is OFF" << endl;
    }
  }

  // debug switch

  debug_ = ps.getUntrackedParameter<bool>("debug", false);

  if ( verbose_ ) {
    if ( debug_ ) {
      cout << " debug switch is ON" << endl;
    } else {
      cout << " debug switch is OFF" << endl;
    }
  }

  // prescaleFactor

  prescaleFactor_ = ps.getUntrackedParameter<int>("prescaleFactor", 1);

  if ( verbose_ ) {
    cout << " prescaleFactor = " << prescaleFactor_ << endl;
  }

  // enableMonitorDaemon switch

  enableMonitorDaemon_ = ps.getUntrackedParameter<bool>("enableMonitorDaemon", false);

  if ( verbose_ ) {
    if ( enableMonitorDaemon_ ) {
      cout << " enableMonitorDaemon switch is ON" << endl;
    } else {
      cout << " enableMonitorDaemon switch is OFF" << endl;
    }
  }

  // prefixME path

  prefixME_ = ps.getUntrackedParameter<string>("prefixME", "");

  if ( verbose_ ) {
    cout << " prefixME path is '" << prefixME_ << "'" << endl;
  }

  // enableCleanup switch

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  if ( verbose_ ) {
    if ( enableCleanup_ ) {
      cout << " enableCleanup switch is ON" << endl;
    } else {
      cout << " enableCleanup switch is OFF" << endl;
    }
  }

  // enableUpdate switch

  enableUpdate_ = ps.getUntrackedParameter<bool>("enableUpdate", false);

  if ( verbose_ ) {
    if ( enableUpdate_ ) {
      cout << " enableUpdate switch is ON" << endl;
    } else {
      cout << " enableUpdate switch is OFF" << endl;
    }
  }

  // DQM Client name

  clientName_ = ps.getUntrackedParameter<string>("clientName", "EcalEndcapMonitorClient");

  if ( enableMonitorDaemon_ ) {

    // DQM Collector hostname

    hostName_ = ps.getUntrackedParameter<string>("hostName", "localhost");

    // DQM Collector port

    hostPort_ = ps.getUntrackedParameter<int>("hostPort", 9090);

    if ( verbose_ ) {
      cout << " Client '" << clientName_ << "' " << endl
           << " Collector on host '" << hostName_ << "'"
           << " on port '" << hostPort_ << "'" << endl;
    }

  }

  // vector of selected Super Modules (Defaults to all 18).

  superModules_.reserve(18);
  for ( unsigned int i = 1; i <= 18; i++ ) superModules_.push_back(i);

  superModules_ = ps.getUntrackedParameter<vector<int> >("superModules", superModules_);

  if ( verbose_ ) {
    cout << " Selected SMs:" << endl;
    for ( unsigned int i = 0; i < superModules_.size(); i++ ) {
      cout << " " << setw(2) << setfill('0') << superModules_[i];
    }
    cout << endl;
  }

  // vector of enabled Clients (defaults)

  enabledClients_.push_back("Integrity");
  enabledClients_.push_back("StatusFlags");
  enabledClients_.push_back("PedestalOnline");
  enabledClients_.push_back("Summary");

  enabledClients_ = ps.getUntrackedParameter<vector<string> >("enabledClients", enabledClients_);

  if ( verbose_ ) {
    cout << " Enabled Clients:" << endl;
    for ( unsigned int i = 0; i < enabledClients_.size(); i++ ) {
      cout << " " << enabledClients_[i];
    }
    cout << endl;
  }

  // set runTypes (use resize() on purpose!)

  runTypes_.resize(30);
  for ( unsigned int i = 0; i < runTypes_.size(); i++ ) runTypes_[i] =  "UNKNOWN";

  runTypes_[EcalDCCHeaderBlock::COSMIC]               = "COSMIC";
  runTypes_[EcalDCCHeaderBlock::BEAMH4]               = "BEAM";
  runTypes_[EcalDCCHeaderBlock::BEAMH2]               = "BEAM";
  runTypes_[EcalDCCHeaderBlock::MTCC]                 = "PHYSICS";
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

    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMIC ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LASER_STD ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LED_STD ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PEDESTAL_STD ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PEDESTAL_OFFSET_SCAN ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::TESTPULSE_MGPA ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::BEAMH4 ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::BEAMH2 ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::MTCC ));

    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMICS_GLOBAL ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PHYSICS_GLOBAL ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMICS_LOCAL ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PHYSICS_LOCAL ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LASER_GAP ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LED_GAP ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::TESTPULSE_GAP ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PEDESTAL_GAP ));

  }

  if ( find(enabledClients_.begin(), enabledClients_.end(), "StatusFlags" ) != enabledClients_.end() ) {

    clients_.push_back( new EEStatusFlagsClient(ps) );
    clientsNames_.push_back( "StatusFlags" );

    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMIC ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LASER_STD ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LED_STD ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PEDESTAL_STD ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PEDESTAL_OFFSET_SCAN ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::TESTPULSE_MGPA ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::BEAMH4 ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::BEAMH2 ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::MTCC ));

    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMICS_GLOBAL ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PHYSICS_GLOBAL ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMICS_LOCAL ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PHYSICS_LOCAL ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LASER_GAP ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LED_GAP ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::TESTPULSE_GAP ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PEDESTAL_GAP ));

  }

  if ( find(enabledClients_.begin(), enabledClients_.end(), "Occupancy" ) != enabledClients_.end() ) {

    clients_.push_back( new EEOccupancyClient(ps) );
    clientsNames_.push_back( "Occupancy" );

    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMIC ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LASER_STD ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LED_STD ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PEDESTAL_STD ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PEDESTAL_OFFSET_SCAN ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::TESTPULSE_MGPA ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::BEAMH4 ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::BEAMH2 ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::MTCC ));

    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMICS_GLOBAL ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PHYSICS_GLOBAL ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMICS_LOCAL ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PHYSICS_LOCAL ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LASER_GAP ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LED_GAP ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::TESTPULSE_GAP ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PEDESTAL_GAP ));

  }

  if ( find(enabledClients_.begin(), enabledClients_.end(), "Cosmic" ) != enabledClients_.end() ) {

    clients_.push_back( new EECosmicClient(ps) );
    clientsNames_.push_back( "Cosmic" );

    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMIC ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LASER_STD ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LED_STD ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PEDESTAL_STD ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::TESTPULSE_MGPA ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::MTCC ));

    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMICS_GLOBAL ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PHYSICS_GLOBAL ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMICS_LOCAL ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PHYSICS_LOCAL ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LASER_GAP ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LED_GAP ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::TESTPULSE_GAP ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PEDESTAL_GAP ));

  }

  if ( find(enabledClients_.begin(), enabledClients_.end(), "Laser" ) != enabledClients_.end() ) {

    clients_.push_back( new EELaserClient(ps) );
    clientsNames_.push_back( "Laser" );

    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMIC ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LASER_STD ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LED_STD ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PEDESTAL_STD ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::TESTPULSE_MGPA ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::BEAMH4 ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::BEAMH2 ));

    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMICS_GLOBAL ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PHYSICS_GLOBAL ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMICS_LOCAL ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PHYSICS_LOCAL ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LASER_GAP ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LED_GAP ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::TESTPULSE_GAP ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PEDESTAL_GAP ));

  }

  if ( find(enabledClients_.begin(), enabledClients_.end(), "Pedestal" ) != enabledClients_.end() ) {

    clients_.push_back( new EEPedestalClient(ps) );
    clientsNames_.push_back( "Pedestal" );

    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMIC ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LASER_STD ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LED_STD ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PEDESTAL_STD ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::TESTPULSE_MGPA ));

    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMICS_GLOBAL ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PHYSICS_GLOBAL ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMICS_LOCAL ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PHYSICS_LOCAL ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LASER_GAP ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LED_GAP ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::TESTPULSE_GAP ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PEDESTAL_GAP ));

  }

  if ( find(enabledClients_.begin(), enabledClients_.end(), "PedestalOnline" ) != enabledClients_.end() ) {

    clients_.push_back( new EEPedestalOnlineClient(ps) );
    clientsNames_.push_back( "PedestalOnline" );

    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMIC ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LASER_STD ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LED_STD ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PEDESTAL_STD ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::TESTPULSE_MGPA ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::BEAMH4 ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::BEAMH2 ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::MTCC ));

    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMICS_GLOBAL ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PHYSICS_GLOBAL ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMICS_LOCAL ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PHYSICS_LOCAL ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LASER_GAP ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LED_GAP ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::TESTPULSE_GAP ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PEDESTAL_GAP ));

  }

  if ( find(enabledClients_.begin(), enabledClients_.end(), "TestPulse" ) != enabledClients_.end() ) {

    clients_.push_back( new EETestPulseClient(ps) );
    clientsNames_.push_back( "TestPulse" );

    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMIC ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LASER_STD ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LED_STD ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PEDESTAL_STD ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::TESTPULSE_MGPA ));

    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMICS_GLOBAL ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PHYSICS_GLOBAL ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMICS_LOCAL ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PHYSICS_LOCAL ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LASER_GAP ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LED_GAP ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::TESTPULSE_GAP ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PEDESTAL_GAP ));

  }

  if ( find(enabledClients_.begin(), enabledClients_.end(), "BeamCalo" ) != enabledClients_.end() ) {

    clients_.push_back( new EEBeamCaloClient(ps) );
    clientsNames_.push_back( "BeamCalo" );

    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::BEAMH4 ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::BEAMH2 ));

  }

  if ( find(enabledClients_.begin(), enabledClients_.end(), "BeamHodo" ) != enabledClients_.end() ) {

    clients_.push_back( new EEBeamHodoClient(ps) );
    clientsNames_.push_back( "BeamHodo" );

    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::BEAMH4 ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::BEAMH2 ));

  }

  if ( find(enabledClients_.begin(), enabledClients_.end(), "TriggerTower" ) != enabledClients_.end() ) {

    clients_.push_back( new EETriggerTowerClient(ps) );
    clientsNames_.push_back( "TriggerTower" );

    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMIC ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::BEAMH4 ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::BEAMH2 ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::MTCC ));

    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMICS_GLOBAL ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PHYSICS_GLOBAL ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMICS_LOCAL ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PHYSICS_LOCAL ));

  }

  if ( find(enabledClients_.begin(), enabledClients_.end(), "Cluster" ) != enabledClients_.end() ) {

    clients_.push_back( new EEClusterClient(ps) );
    clientsNames_.push_back( "Cluster" );

    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMIC ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::BEAMH4 ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::BEAMH2 ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::MTCC ));

    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMICS_GLOBAL ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PHYSICS_GLOBAL ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMICS_LOCAL ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PHYSICS_LOCAL ));

  }

  if ( find(enabledClients_.begin(), enabledClients_.end(), "Timing" ) != enabledClients_.end() ) {

    clients_.push_back( new EETimingClient(ps) );
    clientsNames_.push_back( "Timing" );

    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMIC ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::MTCC ));

    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMICS_GLOBAL ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PHYSICS_GLOBAL ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMICS_LOCAL ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PHYSICS_LOCAL ));

  }

  if ( find(enabledClients_.begin(), enabledClients_.end(), "Led" ) != enabledClients_.end() ) {

    clients_.push_back( new EELedClient(ps) );
    clientsNames_.push_back( "Led" );

    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::COSMIC ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LASER_STD ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LED_STD ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::BEAMH4 ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::BEAMH2 ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PEDESTAL_STD ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::TESTPULSE_MGPA ));

    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PHYSICS_GLOBAL ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PHYSICS_LOCAL ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LASER_GAP ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::LED_GAP ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::TESTPULSE_GAP ));
    clientsRuns_.insert(pair<EEClient*,int>( clients_.back(), EcalDCCHeaderBlock::PEDESTAL_GAP ));

  }

  // define status bits

  clientsStatus_.insert(pair<string,int>( "Integrity",       0 ));
  clientsStatus_.insert(pair<string,int>( "Cosmic",          1 ));
  clientsStatus_.insert(pair<string,int>( "Laser",           2 ));
  clientsStatus_.insert(pair<string,int>( "Pedestal",        3 ));
  clientsStatus_.insert(pair<string,int>( "PedestalOnline",  4 ));
  clientsStatus_.insert(pair<string,int>( "TestPulse",       5 ));
  clientsStatus_.insert(pair<string,int>( "BeamCalo",        6 ));
  clientsStatus_.insert(pair<string,int>( "BeamHodo",        7 ));
  clientsStatus_.insert(pair<string,int>( "TriggerTower",    8 ));
  clientsStatus_.insert(pair<string,int>( "Cluster",         9 ));
  clientsStatus_.insert(pair<string,int>( "Timing",         10 ));
  clientsStatus_.insert(pair<string,int>( "Led",            11 ));
  clientsStatus_.insert(pair<string,int>( "StatusFlags",    12 ));
  clientsStatus_.insert(pair<string,int>( "Occupancy",      13 ));

  if ( find(enabledClients_.begin(), enabledClients_.end(), "Summary" ) != enabledClients_.end() ) {

    summaryClient_ = new EESummaryClient(ps);

  }

  if ( summaryClient_ ) summaryClient_->setFriends(clients_);

  if ( verbose_ ) cout << endl;

}

EcalEndcapMonitorClient::~EcalEndcapMonitorClient() {

  if ( verbose_ ) cout << "Exit ..." << endl;

  for ( unsigned int i=0; i<clients_.size(); i++ ) {
    delete clients_[i];
  }

  if ( summaryClient_ ) delete summaryClient_;

  if ( enableMonitorDaemon_ ) delete mui_;

}

void EcalEndcapMonitorClient::beginJob(const EventSetup &c) {

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

  if ( debug_ ) cout << "EcalEndcapMonitorClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

  current_time_ = time(NULL);
  last_time_update_ = current_time_;
  last_time_db_ = current_time_;

  if ( enableMonitorDaemon_ ) {

    // start DQM user interface instance
    // will attempt to reconnect upon connection problems (w/ a 5-sec delay)

    mui_ = new DQMOldReceiver(hostName_, hostPort_, clientName_, 5);
    dqmStore_ = mui_->getBEInterface();

  } else {

    // get hold of back-end interface

    mui_ = 0;
    dqmStore_ = Service<DQMStore>().operator->();

  }

  if ( ! enableMonitorDaemon_ ) {
    if ( inputFile_.size() != 0 ) {
      if ( dqmStore_ ) {
        dqmStore_->open(inputFile_);
      }
    }
  }

  for ( unsigned int i=0; i<clients_.size(); i++ ) {
    clients_[i]->beginJob(dqmStore_);
  }

  if ( summaryClient_ ) summaryClient_->beginJob(dqmStore_);

  Numbers::initGeometry(c, verbose_);

}

void EcalEndcapMonitorClient::beginRun(void) {

  begin_run_ = true;
  end_run_   = false;

  last_run_  = run_;

  if ( debug_ ) cout << "EcalEndcapMonitorClient: beginRun" << endl;

  jevt_ = 0;

  current_time_ = time(NULL);
  last_time_update_ = current_time_;
  last_time_db_ = current_time_;

  this->setup();

  this->beginRunDb();

  for ( int i=0; i<int(clients_.size()); i++ ) {
    clients_[i]->cleanup();
    bool done = false;
    for ( multimap<EEClient*,int>::iterator j = clientsRuns_.lower_bound(clients_[i]); j != clientsRuns_.upper_bound(clients_[i]); j++ ) {
      if ( runType_ != -1 && runType_ == (*j).second && !done ) {
        done = true;
        clients_[i]->beginRun();
      }
    }
  }

  if ( summaryClient_ ) summaryClient_->beginRun();

}

void EcalEndcapMonitorClient::beginRun(const Run& r, const EventSetup& c) {

  if ( verbose_ ) {
    cout << endl;
    cout << "Standard beginRun() for run " << r.id().run() << endl;
    cout << endl;
  }

  run_ = r.id().run();
  evt_ = 0;

  jevt_ = 0;

}

void EcalEndcapMonitorClient::endJob(void) {

  if ( ! end_run_ ) {

    if ( verbose_ ) {
      cout << endl;
      cout << "Checking last event at endJob() ... " << endl;
      cout << endl;
    }

    forced_update_ = true;
    this->analyze();

    if ( begin_run_ && ! end_run_ ) {

      if ( verbose_ ) {
        cout << endl;
        cout << "Forcing endRun() ... " << endl;
        cout << endl;
      }

      forced_status_ = true;
      this->analyze();
      this->endRun();

    }

  }

  if ( debug_ ) cout << "EcalEndcapMonitorClient: endJob, ievt = " << ievt_ << endl;

  this->cleanup();

  for ( unsigned int i=0; i<clients_.size(); i++ ) {
    clients_[i]->endJob();
  }

  if ( summaryClient_ ) summaryClient_->endJob();

}

void EcalEndcapMonitorClient::endRun(void) {

  begin_run_ = false;
  end_run_   = true;

  if ( debug_ ) cout << "EcalEndcapMonitorClient: endRun, jevt = " << jevt_ << endl;

  if ( subrun_ != -1 ) {

    this->writeDb(true);

    this->endRunDb();

  }

  for ( int i=0; i<int(clients_.size()); i++ ) {
    bool done = false;
    for ( multimap<EEClient*,int>::iterator j = clientsRuns_.lower_bound(clients_[i]); j != clientsRuns_.upper_bound(clients_[i]); j++ ) {
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

void EcalEndcapMonitorClient::endRun(const Run& r, const EventSetup& c) {

  if ( verbose_ ) {
    cout << endl;
    cout << "Standard endRun() for run " << r.id().run() << endl;
    cout << endl;
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

}

void EcalEndcapMonitorClient::beginLuminosityBlock(const LuminosityBlock &l, const EventSetup &c) {

  if ( verbose_ ) {
    cout << endl;
    cout << "Standard beginLuminosityBlock() for run " << l.id().run() << endl;
    cout << endl;
  }

}

void EcalEndcapMonitorClient::endLuminosityBlock(const LuminosityBlock &l, const EventSetup &c) {

  current_time_ = time(NULL);

  if ( verbose_ ) {
    cout << endl;
    cout << "Standard endLuminosityBlock() for run " << l.id().run() << endl;
    cout << endl;
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

  EcalCondDBInterface* econn;

  econn = 0;

  if ( dbName_.size() != 0 ) {
    try {
      if ( verbose_ ) cout << "Opening DB connection with TNS_ADMIN ..." << endl;
      econn = new EcalCondDBInterface(dbName_, dbUserName_, dbPassword_);
      if ( verbose_ ) cout << "done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
      if ( dbHostName_.size() != 0 ) {
        try {
          if ( verbose_ ) cout << "Opening DB connection without TNS_ADMIN ..." << endl;
          econn = new EcalCondDBInterface(dbHostName_, dbName_, dbUserName_, dbPassword_, dbHostPort_);
          if ( verbose_ ) cout << "done." << endl;
        } catch (runtime_error &e) {
          cerr << e.what() << endl;
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
      if ( verbose_ ) cout << "Fetching RunIOV ..." << endl;
//      runiov_ = econn->fetchRunIOV(&runtag, run_);
      runiov_ = econn->fetchRunIOV(location_, run_);
      if ( verbose_ ) cout << "done." << endl;
      foundRunIOV = true;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
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
        if ( verbose_ ) cout << "Inserting RunIOV ..." << endl;
        econn->insertRunIOV(&runiov_);
//        runiov_ = econn->fetchRunIOV(&runtag, run_);
        runiov_ = econn->fetchRunIOV(location_, run_);
        if ( verbose_ ) cout << "done." << endl;
      } catch (runtime_error &e) {
        cerr << e.what() << endl;
        try {
          if ( verbose_ ) cout << "Fetching RunIOV (again) ..." << endl;
//          runiov_ = econn->fetchRunIOV(&runtag, run_);
          runiov_ = econn->fetchRunIOV(location_, run_);
          if ( verbose_ ) cout << "done." << endl;
          foundRunIOV = true;
        } catch (runtime_error &e) {
          cerr << e.what() << endl;
          foundRunIOV = false;
        }
      }
    }

  }

  // end - setup the RunIOV (on behalf of the DAQ)

  if ( verbose_ ) {
    cout << endl;
    cout << "=============RunIOV:" << endl;
    cout << "Run Number:         " << runiov_.getRunNumber() << endl;
    cout << "Run Start:          " << runiov_.getRunStart().str() << endl;
    cout << "Run End:            " << runiov_.getRunEnd().str() << endl;
    cout << "====================" << endl;
    cout << endl;
    cout << "=============RunTag:" << endl;
    cout << "GeneralTag:         " << runiov_.getRunTag().getGeneralTag() << endl;
    cout << "Location:           " << runiov_.getRunTag().getLocationDef().getLocation() << endl;
    cout << "Run Type:           " << runiov_.getRunTag().getRunTypeDef().getRunType() << endl;
    cout << "====================" << endl;
    cout << endl;
  }

  string rt = runiov_.getRunTag().getRunTypeDef().getRunType();
  if ( strcmp(rt.c_str(), "UNKNOWN") == 0 ) {
    runType_ = -1;
  } else {
    for ( unsigned int i = 0; i < runTypes_.size(); i++ ) {
      if ( strcmp(rt.c_str(), runTypes_[i].c_str()) == 0 ) {
        if ( runType_ != int(i) ) {
          if ( verbose_ ) {
            cout << endl;
            cout << "Fixing Run Type to: " << runTypes_[i] << endl;
            cout << endl;
          }
          runType_ = i;
        }
        break;
      }
    }
  }

  if ( maskFile_.size() != 0 ) {
    try {
      if ( verbose_ ) cout << "Fetching masked channels from file ..." << endl;
      EcalErrorMask::readFile(maskFile_, debug_);
      if ( verbose_ ) cout << "done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  } else {
    if ( econn ) {
      try {
        if ( verbose_ ) cout << "Fetching masked channels from DB ..." << endl;
        EcalErrorMask::readDB(econn, &runiov_);
        if ( verbose_ ) cout << "done." << endl;
      } catch (runtime_error &e) {
        cerr << e.what() << endl;
      }
    }
  }

  if ( verbose_ ) cout << endl;

  if ( econn ) {
    try {
      if ( verbose_ ) cout << "Closing DB connection ..." << endl;
      delete econn;
      econn = 0;
      if ( verbose_ ) cout << "done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

  if ( verbose_ ) cout << endl;

}

void EcalEndcapMonitorClient::writeDb(bool flag) {

  subrun_++;

  EcalCondDBInterface* econn;

  econn = 0;

  if ( dbName_.size() != 0 ) {
    try {
      if ( verbose_ ) cout << "Opening DB connection with TNS_ADMIN ..." << endl;
      econn = new EcalCondDBInterface(dbName_, dbUserName_, dbPassword_);
      if ( verbose_ ) cout << "done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
      if ( dbHostName_.size() != 0 ) {
        try {
          if ( verbose_ ) cout << "Opening DB connection without TNS_ADMIN ..." << endl;
          econn = new EcalCondDBInterface(dbHostName_, dbName_, dbUserName_, dbPassword_, dbHostPort_);
          if ( verbose_ ) cout << "done." << endl;
        } catch (runtime_error &e) {
          cerr << e.what() << endl;
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

  // setup the MonIOV

  moniov_.setRunIOV(runiov_);
  moniov_.setSubRunNumber(subrun_);

  if ( enableMonitorDaemon_ || subrun_ > 1 ) {
    moniov_.setSubRunStart(startSubRun);
  } else {
    moniov_.setSubRunStart(runiov_.getRunStart());
  }

  moniov_.setMonRunTag(montag);

  if ( verbose_ ) {
    cout << endl;
    cout << "==========MonRunIOV:" << endl;
    cout << "SubRun Number:      " << moniov_.getSubRunNumber() << endl;
    cout << "SubRun Start:       " << moniov_.getSubRunStart().str() << endl;
    cout << "SubRun End:         " << moniov_.getSubRunEnd().str() << endl;
    cout << "====================" << endl;
    cout << endl;
    cout << "==========MonRunTag:" << endl;
    cout << "GeneralTag:         " << moniov_.getMonRunTag().getGeneralTag() << endl;
    cout << "Monitoring Ver:     " << moniov_.getMonRunTag().getMonVersionDef().getMonitoringVersion() << endl;
    cout << "====================" << endl;
    cout << endl;
  }

  int taskl = 0x0;
  int tasko = 0x0;

  for ( int i=0; i<int(clients_.size()); i++ ) {
    bool done = false;
    for ( multimap<EEClient*,int>::iterator j = clientsRuns_.lower_bound(clients_[i]); j != clientsRuns_.upper_bound(clients_[i]); j++ ) {
      if ( h_ && runType_ != -1 && runType_ == (*j).second && !done ) {
        if ( strcmp(clientsNames_[i].c_str(), "Cosmic") == 0 && runType_ != EcalDCCHeaderBlock::COSMIC && runType_ != EcalDCCHeaderBlock::COSMICS_LOCAL && runType_ != EcalDCCHeaderBlock::COSMICS_GLOBAL && runType_ != EcalDCCHeaderBlock::PHYSICS_GLOBAL && runType_ != EcalDCCHeaderBlock::PHYSICS_LOCAL && h_->GetBinContent(2+EcalDCCHeaderBlock::COSMIC) == 0 && h_->GetBinContent(2+EcalDCCHeaderBlock::COSMICS_LOCAL) == 0 && h_->GetBinContent(2+EcalDCCHeaderBlock::COSMICS_GLOBAL) == 0 && h_->GetBinContent(2+EcalDCCHeaderBlock::PHYSICS_GLOBAL) == 0 && h_->GetBinContent(2+EcalDCCHeaderBlock::PHYSICS_LOCAL) == 0 ) continue;
        if ( strcmp(clientsNames_[i].c_str(), "Laser") == 0 && runType_ != EcalDCCHeaderBlock::LASER_STD && runType_ != EcalDCCHeaderBlock::LASER_GAP && h_->GetBinContent(2+EcalDCCHeaderBlock::LASER_STD) == 0 && h_->GetBinContent(2+EcalDCCHeaderBlock::LASER_GAP) == 0 ) continue;
        if ( strcmp(clientsNames_[i].c_str(), "Led") == 0 && runType_ != EcalDCCHeaderBlock::LED_STD && runType_ != EcalDCCHeaderBlock::LED_GAP && h_->GetBinContent(2+EcalDCCHeaderBlock::LED_STD) == 0 && h_->GetBinContent(2+EcalDCCHeaderBlock::LED_GAP) == 0 ) continue;
        if ( strcmp(clientsNames_[i].c_str(), "Pedestal") == 0 && runType_ != EcalDCCHeaderBlock::PEDESTAL_STD && runType_ != EcalDCCHeaderBlock::PEDESTAL_GAP && h_->GetBinContent(2+EcalDCCHeaderBlock::PEDESTAL_STD) == 0 && h_->GetBinContent(2+EcalDCCHeaderBlock::PEDESTAL_GAP) == 0 ) continue;
        if ( strcmp(clientsNames_[i].c_str(), "TestPulse") == 0 && runType_ != EcalDCCHeaderBlock::TESTPULSE_MGPA && runType_ != EcalDCCHeaderBlock::TESTPULSE_GAP && h_->GetBinContent(2+EcalDCCHeaderBlock::TESTPULSE_MGPA) == 0 && h_->GetBinContent(2+EcalDCCHeaderBlock::TESTPULSE_GAP) == 0 ) continue;
        done = true;
        if ( verbose_ ) {
          if ( econn ) {
            cout << " Writing " << clientsNames_[i] << " results to DB " << endl;
            cout << endl;
          }
        }
        bool status;
        if ( clients_[i]->writeDb(econn, &runiov_, &moniov_, status, flag) ) {
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
        cout << " Task output for " << clientsNames_[i] << " = "
             << ((tasko >> clientsStatus_[clientsNames_[i]]) & 0x1) << endl;
        cout << endl;
      }
    }
  }

  bool status;
  if ( summaryClient_ ) summaryClient_->writeDb(econn, &runiov_, &moniov_, status, flag);

  EcalLogicID ecid;
  MonRunDat md;
  map<EcalLogicID, MonRunDat> dataset;

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
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

  if ( econn ) {
    try {
      if ( verbose_ ) cout << "Inserting MonRunDat ..." << endl;
      econn->insertDataSet(&dataset, &moniov_);
      if ( verbose_ ) cout << "done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

  if ( econn ) {
    try {
      if ( verbose_ ) cout << "Closing DB connection ..." << endl;
      delete econn;
      econn = 0;
      if ( verbose_ ) cout << "done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

  if ( verbose_ ) cout << endl;

}

void EcalEndcapMonitorClient::endRunDb(void) {

  EcalCondDBInterface* econn;

  econn = 0;

  if ( dbName_.size() != 0 ) {
    try {
      if ( verbose_ ) cout << "Opening DB connection with TNS_ADMIN ..." << endl;
      econn = new EcalCondDBInterface(dbName_, dbUserName_, dbPassword_);
      if ( verbose_ ) cout << "done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
      if ( dbHostName_.size() != 0 ) {
        try {
          if ( verbose_ ) cout << "Opening DB connection without TNS_ADMIN ..." << endl;
          econn = new EcalCondDBInterface(dbHostName_, dbName_, dbUserName_, dbPassword_, dbHostPort_);
          if ( verbose_ ) cout << "done." << endl;
        } catch (runtime_error &e) {
          cerr << e.what() << endl;
        }
      }
    }
  }

  EcalLogicID ecid;
  RunDat rd;
  map<EcalLogicID, RunDat> dataset;

  float nevt = -1.;

  if ( h_ ) nevt = h_->GetSumOfWeights();

  rd.setNumEvents(int(nevt));

  // fetch the RunDat from the DB

  bool foundRunDat = false;

  if ( econn ) {
    try {
      if ( verbose_ ) cout << "Fetching RunDat ..." << endl;
      econn->fetchDataSet(&dataset, &runiov_);
      if ( verbose_ ) cout << "done." << endl;
      foundRunDat = true;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
      foundRunDat = false;
    }
  }

  // begin - setup the RunDat (on behalf of the DAQ)

  if ( ! foundRunDat ) {

    if ( econn ) {
      try {
        ecid = LogicID::getEcalLogicID("EE");
        dataset[ecid] = rd;
      } catch (runtime_error &e) {
        cerr << e.what() << endl;
      }
    }

    if ( econn ) {
      try {
        if ( verbose_ ) cout << "Inserting RunDat ..." << endl;
        econn->insertDataSet(&dataset, &runiov_);
        if ( verbose_ ) cout << "done." << endl;
      } catch (runtime_error &e) {
        cerr << e.what() << endl;
      }
    }

  }

  // end - setup the RunDat (on behalf of the DAQ)

  if ( econn ) {
    try {
      if ( verbose_ ) cout << "Closing DB connection ..." << endl;
      delete econn;
      econn = 0;
      if ( verbose_ ) cout << "done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

}

void EcalEndcapMonitorClient::analyze(void) {

  current_time_ = time(NULL);

  ievt_++;
  jevt_++;

  if ( debug_ ) cout << "EcalEndcapMonitorClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;

  // update MEs (online mode)
  if ( enableUpdate_ ) {
    if ( enableMonitorDaemon_ ) mui_->doMonitoring();
  }

  MonitorElement* me;
  string s;

  me = dqmStore_->get(prefixME_ + "/EcalInfo/STATUS");
  if ( me ) {
    status_ = "unknown";
    s = me->valueString();
    if ( strcmp(s.c_str(), "i=0") == 0 ) status_ = "begin-of-run";
    if ( strcmp(s.c_str(), "i=1") == 0 ) status_ = "running";
    if ( strcmp(s.c_str(), "i=2") == 0 ) status_ = "end-of-run";
    if ( debug_ ) cout << "Found '" << prefixME_ << "/EcalInfo/STATUS'" << endl;
  }

  if ( inputFile_.size() != 0 ) {
    if ( ievt_ == 1 ) {
      if ( verbose_ ) {
        cout << endl;
        cout << " Reading DQM from file, forcing 'begin-of-run'" << endl;
        cout << endl;
      }
      status_ = "begin-of-run";
    }
  }

  int ecal_run = -1;
  me = dqmStore_->get(prefixME_ + "/EcalInfo/RUN");
  if ( me ) {
    s = me->valueString();
    sscanf(s.c_str(), "i=%d", &ecal_run);
    if ( debug_ ) cout << "Found '" << prefixME_ << "/EcalInfo/RUN'" << endl;
  }

  int ecal_evt = -1;
  me = dqmStore_->get(prefixME_ + "/EcalInfo/EVT");
  if ( me ) {
    s = me->valueString();
    sscanf(s.c_str(), "i=%d", &ecal_evt);
    if ( debug_ ) cout << "Found '" << prefixME_ << "/EcalInfo/EVT'" << endl;
  }

  me = dqmStore_->get(prefixME_ + "/EcalInfo/EVTTYPE");
  h_ = UtilsClient::getHisto<TH1F*>( me, cloneME_, h_ );

  me = dqmStore_->get(prefixME_ + "/EcalInfo/RUNTYPE");
  if ( me ) {
    s = me->valueString();
    sscanf(s.c_str(), "i=%d", &evtType_);
    if ( runType_ == -1 ) runType_ = evtType_;
    if ( debug_ ) cout << "Found '" << prefixME_ << "/EcalInfo/RUNTYPE'" << endl;
  }

  // if the run number from the Event is less than zero,
  // use the run number from the ECAL DCC header
  if ( run_ <= 0 ) run_ = ecal_run;

  if ( run_ != -1 && evt_ != -1 && runType_ != -1 ) {
    if ( ! mergeRuns_ && run_ != last_run_ ) forced_update_ = true;
  }

  bool update = ( prescaleFactor_ != 1               ) ||
                ( jevt_ <   10                       ) ||
                ( jevt_ <  100 && jevt_ %    10 == 0 ) ||
                ( jevt_ < 1000 && jevt_ %   100 == 0 ) ||
                (                 jevt_ % 10000 == 0 );
 
  if ( update || strcmp(status_.c_str(), "begin-of-run") == 0 || strcmp(status_.c_str(), "end-of-run") == 0 || forced_update_ ) {

    if ( verbose_ ) {
      cout << " RUN status = \"" << status_ << "\"" << endl;
      cout << "   CMS  run/event number = " << run_ << "/" << evt_ << endl;
      cout << "   EE run/event number = " << ecal_run << "/" << ecal_evt << endl;
      cout << "   EE location = " << location_ << endl;
      cout << "   EE run/event type = " << this->getRunType() << "/" << ( evtType_ == -1 ? "UNKNOWN" : runTypes_[evtType_] ) << flush;

      if ( h_ ) {
        if ( h_->GetSumOfWeights() != 0 ) {
          cout << " ( " << flush;
          for ( unsigned int i = 0; i < runTypes_.size(); i++ ) {
            if ( strcmp(runTypes_[i].c_str(), "UNKNOWN") != 0 && h_->GetBinContent(2+i) != 0 ) {
              string s = runTypes_[i];
              transform( s.begin(), s.end(), s.begin(), (int(*)(int))tolower );
              cout << s << " ";
            }
          }
          cout << ")" << flush;
        }
      }
      cout << endl;
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

      bool update = ( prescaleFactor_ != 1 || jevt_ < 3 || jevt_ % 1000 == 0 );

      if ( update || strcmp(status_.c_str(), "begin-of-run") == 0 || strcmp(status_.c_str(), "end-of-run") == 0 || forced_update_ ) {

        for ( int i=0; i<int(clients_.size()); i++ ) {
          bool done = false;
          for ( multimap<EEClient*,int>::iterator j = clientsRuns_.lower_bound(clients_[i]); j != clientsRuns_.upper_bound(clients_[i]); j++ ) {
            if ( runType_ != -1 && runType_ == (*j).second && !done ) {
              done = true;
              clients_[i]->analyze();
            }
          }
        }

        if ( summaryClient_ ) summaryClient_->analyze();

      }

      forced_update_ = false;

      if ( dbUpdateTime_ > 0 ) {
        if ( (current_time_ - last_time_db_) > 60 * dbUpdateTime_ ) {
          if ( runType_ == EcalDCCHeaderBlock::COSMIC ||
               runType_ == EcalDCCHeaderBlock::COSMICS_GLOBAL ||
               runType_ == EcalDCCHeaderBlock::PHYSICS_GLOBAL ||
               runType_ == EcalDCCHeaderBlock::COSMICS_LOCAL ||
               runType_ == EcalDCCHeaderBlock::PHYSICS_LOCAL ||
               runType_ == EcalDCCHeaderBlock::BEAMH2 ||
               runType_ == EcalDCCHeaderBlock::BEAMH4 ) this->writeDb(false);
          last_time_db_ = current_time_;
        }
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
              cout << endl;
              cout << " Old run has finished, issuing endRun() ... " << endl;
              cout << endl;
            }

            // end old_run_
            run_ = old_run_;

            forced_status_ = false;
            this->endRun();

          }

          if ( ! begin_run_ ) {

            if ( verbose_ ) {
              cout << endl;
              cout << " New run has started, issuing beginRun() ... " << endl;
              cout << endl;
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
            cout << endl;
            cout << "Forcing beginRun() ... NOW !" << endl;
            cout << endl;
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
            cout << endl;
            cout << "Forcing beginRun() ... NOW !" << endl;
            cout << endl;
          }

          forced_status_ = true;
          this->beginRun();

        }

      }

    }

  }

  // END: run-time fixes for missing state transitions

}

void EcalEndcapMonitorClient::analyze(const Event &e, const EventSetup &c) {

  run_ = e.id().run();
  evt_ = e.id().event();

  if ( prescaleFactor_ > 0 ) {
    if ( jevt_ % prescaleFactor_ == 0 ) this->analyze();
  }

}

void EcalEndcapMonitorClient::softReset(bool flag) {

   for ( int i=0; i<int(clients_.size()); i++ ) {
     bool done = false;
     for ( multimap<EEClient*,int>::iterator j = clientsRuns_.lower_bound(clients_[i]); j != clientsRuns_.upper_bound(clients_[i]); j++ ) {
       if ( runType_ != -1 && runType_ == (*j).second && !done ) {
         done = true;
         clients_[i]->softReset(flag);
       }
     }
   }

   summaryClient_->softReset(flag);

}


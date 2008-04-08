// Last commit: $Id: SiStripConfigDb.h,v 1.50 2008/03/28 15:16:15 bainbrid Exp $

#ifndef OnlineDB_SiStripConfigDb_SiStripConfigDb_h
#define OnlineDB_SiStripConfigDb_SiStripConfigDb_h

#define DATABASE //@@ very necesssary!

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripDbParams.h"
#include "DeviceFactory.h"
#include "boost/cstdint.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <ostream>
#include <vector>
#include <string>
#include <map>

//#define USING_NEW_DATABASE_MODEL
//#define USING_DATABASE_CACHE

#ifdef USING_DATABASE_CACHE
#include "DbClient.h"
#else
class DbClient;
#endif

/*
  - Remaining work:
  - run number, o2o
  - multi-partitions (multi-download, inhibit upload?)
  - calibrations data-handling
  - psu mapping
*/

#ifdef USING_NEW_DATABASE_MODEL
namespace sistrip {
  static const uint16_t FEC_CRATE_OFFSET =  0; //@@ temporary
  static const uint16_t FEC_RING_OFFSET  =  0; //@@ temporary
}
#else
namespace sistrip {
  static const uint16_t FEC_CRATE_OFFSET =  1; //@@ temporary
  static const uint16_t FEC_RING_OFFSET  =  1; //@@ temporary
}
#endif

// Friend class
namespace cms { class SiStripO2O; }

/**	
   \class SiStripConfigDb
   \brief An interface class to the DeviceFactory
   \author R.Bainbridge
*/
class SiStripConfigDb {
  
  
 public:


  // ---------- Constructors, destructors ----------

  
  /** Constructor when using the "service" mode, which takes as an
      argument a ParameterSet (containing the database connection
      parameters). */
  SiStripConfigDb( const edm::ParameterSet&,
		   const edm::ActivityRegistry& );
  
  /** Default destructor. */
  ~SiStripConfigDb();
  

  // ---------- PROTECTED INTERFACE ----------
  
  
 protected:
  
  
  /*
    Access to the configuration database is reserved solely for the
    commissioning (database) client and the online-to-offline transfer
    tool. If you wish to use this interface to the configuration
    database, then please contact one of the package administrators.
  */
  
  // ESSources and O2O
  friend class SiStripFedCablingBuilderFromDb;
  friend class SiStripPedestalsBuilderFromDb;
  friend class SiStripNoiseBuilderFromDb;
  friend class cms::SiStripO2O;
  
  // Commissioning clients
  friend class SiStripCommissioningDbClient;
  friend class SiStripCommissioningOfflineDbClient;
  friend class CommissioningHistosUsingDb;
  friend class FastFedCablingHistosUsingDb;
  friend class FedCablingHistosUsingDb;
  friend class ApvTimingHistosUsingDb;
  friend class OptoScanHistosUsingDb;
  friend class PedestalsHistosUsingDb;
  friend class PedsOnlyHistosUsingDb;
  friend class NoiseHistosUsingDb;
  friend class VpspScanHistosUsingDb;
  friend class LatencyHistosUsingDb;
  friend class FineDelayHistosUsingDb;
  friend class CalibrationHistosUsingDb;

  // Utility and tests
  friend class PopulateConfigDb;
  friend class testSiStripConfigDb;


  // ---------- Typedefs and structs ----------


  typedef std::vector<deviceDescription*> DeviceDescriptions;

  typedef std::vector<Fed9U::Fed9UDescription*> FedDescriptions;

#ifdef USING_NEW_DATABASE_MODEL

  typedef std::vector<ConnectionDescription*> FedConnections;

  typedef CommissioningAnalysisDescription::commissioningType AnalysisType;

#else

  //@@ TO BE DEPRECATED
  class CommissioningAnalysisDescription;
  
  typedef std::vector<FedChannelConnectionDescription*> FedConnections;

#endif

  typedef CommissioningAnalysisDescription AnalysisDescription;

  typedef std::vector<AnalysisDescription*> AnalysisDescriptions;

  /** Key is DCU id. */
  typedef Sgi::hash_map<unsigned long,TkDcuInfo*> DcuDetIdMap;
  
  /** Class that holds addresses that uniquely identify a hardware
      component within the control system. */
  class DeviceAddress { 
  public:
    DeviceAddress();
    void reset();
    uint16_t fecCrate_; 
    uint16_t fecSlot_;
    uint16_t fecRing_;
    uint16_t ccuAddr_;
    uint16_t ccuChan_;
    uint16_t lldChan_;
    uint16_t i2cAddr_;
    uint16_t fedId_;
    uint16_t feUnit_;
    uint16_t feChan_;
  };
  
  
  // ---------- Connection and local cache ----------

  
  /** Establishes connection to DeviceFactory API. */
  void openDbConnection();
  
  /** Closes connection to DeviceFactory API. */
  void closeDbConnection();

  /** Returns DB connection parameters. */
  inline const SiStripDbParams& dbParams() const;
  
  /** Returns whether using database or xml files. */
  inline const bool& usingDb() const;
  
  /** Returns pointer to DeviceFactory API, with check if NULL. */
  DeviceFactory* const deviceFactory( std::string method_name = "" ) const;

  /** Returns pointer to DeviceFactory API, with check if NULL. */
  DbClient* const databaseCache( std::string method_name = "" ) const;
  

  // ---------- FEC / Front-End devices ---------- 

  /** Returns descriptions for given device type (which can be one
      of the following: APV25, APVMUX, DCU, LASERDRIVER, PLL ). */
  const DeviceDescriptions& getDeviceDescriptions( const enumDeviceType& );
  
  /** Fills local cache with all device descriptions from DB/xml. */
  const DeviceDescriptions& getDeviceDescriptions(); 
  
  /** Uploads all device descriptions in cache to DB/xml. */
  void uploadDeviceDescriptions( bool new_major_version = true ); 
  
  /** Creates device descriptions based on FEC cabling. */
  void createDeviceDescriptions( const SiStripFecCabling& );
  
  /** Extracts unique hardware address of device from description. */
  DeviceAddress deviceAddress( const deviceDescription& ); //@@ uses temp offsets!
  

  // ---------- FED descriptions ----------


  /** Fills local cache with FED descriptions from DB/xml. */
  const FedDescriptions& getFedDescriptions();
  
  /** Uploads FED descriptions to DB/xml. */
  void uploadFedDescriptions( bool new_major_version = true );
  
  /** Create "dummy" FED descriptions based on FED cabling. */
  void createFedDescriptions( const SiStripFecCabling& );
  
  /** Extracts FED ids from FED descriptions. */
  const std::vector<uint16_t>& getFedIds();
  
  /** Indicates if strip info is enabled/disabled within FED descs. */
  inline const bool& usingStrips() const;
  
  /** Enable/disable strip info within FED descriptions. */
  inline void usingStrips( bool );
  

  // ---------- FED connections ----------


  /** Fills local cache with connection descriptions from DB/xml. */
  const FedConnections& getFedConnections();
  
  /** Uploads FED-FEC connections to DB/xml. */
  void uploadFedConnections( bool new_major_version = true );
  
  /** Creates "dummy" FED connections based on FEC cabling. */
  void createFedConnections( const SiStripFecCabling& );
  

  // ---------- Commissioning analyses ---------- 
  
#ifdef USING_NEW_DATABASE_MODEL

  /** Returns analysis descriptions for given analysis type:
      T_UNKNOWN, 
      T_ANALYSIS_FASTFEDCABLING, 
      T_ANALYSIS_TIMING,
      T_ANALYSIS_OPTOSCAN, 
      T_ANALYSIS_VPSPSCAN,
      T_ANALYSIS_PEDESTAL, 
      T_ANALYSIS_APVLATENCY, 
      T_ANALYSIS_FINEDELAY,
      T_ANALYSIS_CALIBRATION.
  */
  const AnalysisDescriptions& getAnalysisDescriptions( const AnalysisType& );
  
  /** Uploads all analysis descriptions in cache to DB/xml. Must be
      called AFTER the upload of any hardware parameters. */
  void uploadAnalysisDescriptions( bool use_as_calibrations_for_physics = false ); 

  /** Appends analysis descriptions to internal cache. */
  void createAnalysisDescriptions( AnalysisDescriptions& );
  
  /** Creates analysis descriptions based on FEC cabling. */
  void createAnalysisDescriptions( const SiStripFecCabling& ) {;}
  
  /** Extracts unique hardware address of device from description. */
  DeviceAddress deviceAddress( const AnalysisDescription& ); //@@ uses temp offsets!

  /** */
  std::string analysisType( const AnalysisType& analysis_type ) const;

#endif


  // ---------- DCU-DetId info ----------


  /** Returns the DcuId-DetId map. If the local cache is empty, it
      retrieves the DcuId-DetId map from the DB/xml file. */
  const DcuDetIdMap& getDcuDetIdMap();
  
  /** Uploads the contents of the local cache to DB/xml file. */
  void uploadDcuDetIdMap();

  
  // ---------- Miscellaneous ----------

  
  /** Creates "dummy" descriptions based on FEC cabling. */
  void createPartition( const std::string& partition_name,
			const SiStripFecCabling&,
			const DcuDetIdMap& ); 
  
  
 private:
  

  // ---------- Private methods ----------

  
  /** */
  void usingDatabase();

  /** */
  void usingDatabaseCache();
  
  /** */
  void usingXmlFiles();
  
  /** Handles exceptions thrown by FEC and FED software. */
  void handleException( const std::string& method_name,
			const std::string& extra_info = "" );// throw (cms::Exception);
  
  /** Checks whether file at "path" exists or not. */
  bool checkFileExists( const std::string& path );
  
  /** Returns device identifier based on device type. */
  std::string deviceType( const enumDeviceType& device_type ) const;

  
  // ---------- Database connection, partitions and versions ----------

  
  /** Pointer to the DeviceFactory API. */
  DeviceFactory* factory_; 

  /** Pointer to the DbClient class. */
  DbClient* dbCache_; 

  /** Instance of struct that holds all DB connection parameters. */
  SiStripDbParams dbParams_;


  // ---------- Local cache ----------


  /** Vector of descriptions for all FEC devices (including DCUs). */
  DeviceDescriptions devices_;

  /** Fed9U descriptions. */
  FedDescriptions feds_;

  /** FED-FEC connection descriptions. */
  FedConnections connections_;

#ifdef USING_NEW_DATABASE_MODEL

  /** Vector of analysis descriptions for all commissioning runs. */
  AnalysisDescriptions analyses_;

#endif
 
  /** TkDcuInfo objects, containing DcuId-DetId map. */
  DcuDetIdMap dcuDetIdMap_;
  
  /** FED ids. */ 
  std::vector<uint16_t> fedIds_;


  // ---------- Miscellaneous ----------

  
  /** Switch to enable/disable transfer of strip information. */
  bool usingStrips_;

  /** */
  bool openConnection_;
  
  /** Static counter of instances of this class. */
  static uint32_t cntr_;

  static bool allowCalibUpload_;
  
};


// ---------- Inline methods ----------


/** Returns DB connection parameters. */
const SiStripDbParams& SiStripConfigDb::dbParams() const { return dbParams_; }

/** Indicates whether DB (true) or XML files (false) are used. */
const bool& SiStripConfigDb::usingDb() const { return dbParams_.usingDb_; }

/** Indicates whether FED strip info is uploaded/downloaded. */
const bool& SiStripConfigDb::usingStrips() const { return usingStrips_; }

/** Switches on/off of upload/download for FED strip info. */
void SiStripConfigDb::usingStrips( bool using_strips ) { usingStrips_ = using_strips; }

#endif // OnlineDB_SiStripConfigDb_SiStripConfigDb_h

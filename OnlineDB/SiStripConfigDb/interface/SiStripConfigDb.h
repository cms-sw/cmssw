// Last commit: $Id: SiStripConfigDb.h,v 1.45 2008/03/17 18:57:25 bainbrid Exp $

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
#include "DeviceFactory.h"
#include "boost/cstdint.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <ostream>
#include <vector>
#include <string>
#include <map>

#define USING_NEW_DATABASE_MODEL
#define USING_DATABASE_CACHE

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

  /** Constructor when using the configuration database, which takes
      as arguments the database connection parameters. DO NOT USE: TO
      BE DEPRECATED!!! */
  SiStripConfigDb( std::string confdb, 
		   std::string db_partition,
		   uint32_t    db_major_vers = 0,
		   uint32_t    db_minor_vers = 0 ); 

  /** Constructor when using the configuration database, which takes
      as arguments the database connection parameters. DO NOT USE: TO
      BE DEPRECATED!!! */
  SiStripConfigDb( std::string db_user, 
		   std::string db_passwd, 
		   std::string db_path,
		   std::string db_partition,
		   uint32_t    db_major_vers = 0,
		   uint32_t    db_minor_vers = 0 ); 
  
  /** Constructor when using xml files, which takes as arguments the
      paths to the various input (and output) xml files. DO NOT USE:
      TO BE DEPRECATED!!! */
  SiStripConfigDb( std::string input_module_xml,
		   std::string input_dcuinfo_xml,
		   std::vector<std::string> input_fec_xml,
		   std::vector<std::string> input_fed_xml,
		   std::string output_module_xml = "/tmp/module.xml",
		   std::string output_dcuinfo_xml = "/tmp/dcuinfo.xml",
		   std::string output_fec_xml = "/tmp/fec.xml",
		   std::string output_fed_xml = "/tmp/fed.xml" );
  
  /** Default destructor. */
  ~SiStripConfigDb();
  

  // ---------- Typedefs ----------


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
  
  
  // ---------- Structs and enums ----------
  
  
  /** 
   * Container class for database connection parameters: 
   * Connection params: usingDB flag, confdb, user, passwd and path strings.
   * Partition info: partition name and major, minor versions.
   * Input XML files: for modules, DCU-DetId map (dcuinfo.xml), FECs and FEDs.
   * Output XML files: as for input XML files.
   */
  class DbParams { 
  public:
    // Constructor and methods
    DbParams();
    ~DbParams();
    void print( std::stringstream& ) const; 
    void reset(); 
    void setParams( const edm::ParameterSet& );
    void confdb( const std::string& );
    void confdb( const std::string& user,
		 const std::string& passwd,
		 const std::string& path );
    // Public member data 
    bool usingDb_;
    std::string confdb_;
    std::string user_;
    std::string passwd_;
    std::string path_;
    std::string partition_; 
    bool usingDbCache_;
    std::string sharedMemory_;
    uint32_t runNumber_;
    uint32_t cabMajor_;
    uint32_t cabMinor_;
    uint32_t fedMajor_;
    uint32_t fedMinor_;
    uint32_t fecMajor_;
    uint32_t fecMinor_;
    uint32_t calMajor_;
    uint32_t calMinor_;
    uint32_t dcuMajor_;
    uint32_t dcuMinor_;
    sistrip::RunType runType_;
    bool force_;
    std::string inputModuleXml_;
    std::string inputDcuInfoXml_;
    std::vector<std::string> inputFecXml_;
    std::vector<std::string> inputFedXml_;
    std::string inputDcuConvXml_;
    std::string outputModuleXml_;
    std::string outputDcuInfoXml_;
    std::string outputFecXml_;
    std::string outputFedXml_;
    std::string tnsAdmin_;
  };
  
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


  // ---------- PRIVATE INTERFACE ----------

  
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

 protected:
  

  // ---------- Connection and local cache ----------

  
  /** Establishes connection to DeviceFactory API. */
  void openDbConnection();
  
  /** Closes connection to DeviceFactory API. */
  void closeDbConnection();

  /** Returns DB connection parameters. */
  inline const DbParams& dbParams() const;
  
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
  DbParams dbParams_;


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
const SiStripConfigDb::DbParams& SiStripConfigDb::dbParams() const { return dbParams_; }

/** Indicates whether DB (true) or XML files (false) are used. */
const bool& SiStripConfigDb::usingDb() const { return dbParams_.usingDb_; }

/** Indicates whether FED strip info is uploaded/downloaded. */
const bool& SiStripConfigDb::usingStrips() const { return usingStrips_; }

/** Switches on/off of upload/download for FED strip info. */
void SiStripConfigDb::usingStrips( bool using_strips ) { usingStrips_ = using_strips; }

/** Debug printout for DbParams class. */
std::ostream& operator<< ( std::ostream&, const SiStripConfigDb::DbParams& );

#endif // OnlineDB_SiStripConfigDb_SiStripConfigDb_h

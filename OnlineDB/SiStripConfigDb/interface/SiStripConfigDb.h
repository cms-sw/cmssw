// Last commit: $Id: SiStripConfigDb.h,v 1.58 2008/04/23 12:17:48 bainbrid Exp $

#ifndef OnlineDB_SiStripConfigDb_SiStripConfigDb_h
#define OnlineDB_SiStripConfigDb_SiStripConfigDb_h

#define DATABASE // Needed by DeviceFactory API! Do not comment!
//#define USING_NEW_DATABASE_MODEL

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "DataFormats/Common/interface/MapOfVectors.h"
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

#ifdef USING_NEW_DATABASE_MODEL
#include "DbClient.h"
#else
class DbClient;
#endif

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


  // ---------- Typedefs ----------


#ifdef USING_NEW_DATABASE_MODEL
  typedef ConnectionDescription FedConnection;
#else
  typedef FedChannelConnectionDescription FedConnection;
#endif
  typedef edm::MapOfVectors<std::string,FedConnection*> FedConnections;
  
  typedef deviceDescription DeviceDescription;
  typedef edm::MapOfVectors<std::string,DeviceDescription*> DeviceDescriptions;
  typedef std::pair< std::vector<DeviceDescription*>::iterator, std::vector<DeviceDescription*>::iterator > DeviceDescriptionsRange;

  typedef Fed9U::Fed9UDescription FedDescription;
  typedef std::vector<FedDescription*> FedDescriptions;
  
  typedef TkDcuInfo DcuDetId; 
  typedef Sgi::hash_map<unsigned long,DcuDetId*> DcuDetIdMap; //@@ Key is DCU id

#ifdef USING_NEW_DATABASE_MODEL
  typedef CommissioningAnalysisDescription::commissioningType AnalysisType;
#else
  class CommissioningAnalysisDescription;
#endif
  typedef CommissioningAnalysisDescription AnalysisDescription;
  typedef std::vector<AnalysisDescription*> AnalysisDescriptions;


  // ---------- Useful structs ----------


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

  
  // ---------- Connection and useful methods ----------

  
  /** Establishes connection to DeviceFactory API. */
  void openDbConnection();
  
  /** Closes connection to DeviceFactory API. */
  void closeDbConnection();

  /** Returns database connection parameters. */
  inline const SiStripDbParams& dbParams() const;
  
  /** Returns whether using database or xml files. */
  inline const bool& usingDb() const;
  
  /** Returns pointer to DeviceFactory API, with check if NULL. */
  DeviceFactory* const deviceFactory( std::string method_name = "" ) const;

  /** Returns pointer to DeviceFactory API, with check if NULL. */
  DbClient* const databaseCache( std::string method_name = "" ) const;
  
  /** Updates versions for specified partition. */
  void useVersions( SiStripPartition& );
  
  /** Updates versions for specified partition and run. */
  void useRunNumber( SiStripPartition& );
  
  
  // ---------- FED connections ----------


  /** Returns local cache (just for given partition if specified). */
  FedConnections::range getFedConnections( std::string partition = "" );

  /** Add to local cache (just for given partition if specified). */
  void addFedConnections( std::string partition, std::vector<FedConnection*>& );
  
  /** Uploads to database (just for given partition if specified). */
  void uploadFedConnections( std::string partition = "" );
  
  /** Clears local cache (just for given partition if specified). */
  void clearFedConnections( std::string partition = "" );

  /** Prints local cache (just for given partition if specified). */
  void printFedConnections( std::string partition = "" );
  

  // ---------- FEC / Front-End devices ---------- 

  
  /** Returns local cache (just for given partition if specified). */
  DeviceDescriptions::range getDeviceDescriptions( std::string partition = "" ); 

  /** Returns (pair of iterators to) descriptions of given type. */
  /** (APV25, APVMUX, DCU, LASERDRIVER, PLL). */
  DeviceDescriptionsRange getDeviceDescriptions( const enumDeviceType& type,
						 std::string partition = "" );
  
  /** Adds to local cache (just for given partition if specified). */
  void addDeviceDescriptions( std::string partition, std::vector<DeviceDescription*>& );
  
  /** Uploads to database (just for given partition if specified). */
  void uploadDeviceDescriptions( std::string partition = "" ); 
  
  /** Clears local cache (just for given partition if specified). */
  void clearDeviceDescriptions( std::string partition = "" ); 
  
  /** Prints local cache (just for given partition if specified). */
  void printDeviceDescriptions( std::string partition = "" ); 
  
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
  

  // ---------- DCU-DetId info ----------


  /** Returns the DcuId-DetId map. If the local cache is empty, it
      retrieves the DcuId-DetId map from the DB/xml file. */
  const DcuDetIdMap& getDcuDetIdMap();
  
  /** Uploads the contents of the local cache to DB/xml file. */
  void uploadDcuDetIdMap();
  

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


  // ---------- Local cache of vectors ----------


  /** FED-FEC connection descriptions. */
  FedConnections connections_;
  
  /** Device descriptions (including DCUs). */
  DeviceDescriptions devices_;

  /** Cache for devices of given type. */
  std::vector<DeviceDescription*> typedDevices_;

  /** Fed9U descriptions. */
  FedDescriptions feds_;
  
#ifdef USING_NEW_DATABASE_MODEL

  /** Analysis descriptions for given commissioning run. */
  AnalysisDescriptions analyses_;

#endif
 
  /** DcuId-DetId map (map of TkDcuInfo objects). */
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


/** Returns database connection parameters. */
const SiStripDbParams& SiStripConfigDb::dbParams() const { return dbParams_; }

/** Indicates whether DB (true) or XML files (false) are used. */
const bool& SiStripConfigDb::usingDb() const { return dbParams_.usingDb_; }

/** Indicates whether FED strip info is uploaded/downloaded. */
const bool& SiStripConfigDb::usingStrips() const { return usingStrips_; }

/** Switches on/off of upload/download for FED strip info. */
void SiStripConfigDb::usingStrips( bool using_strips ) { usingStrips_ = using_strips; }


#endif // OnlineDB_SiStripConfigDb_SiStripConfigDb_h

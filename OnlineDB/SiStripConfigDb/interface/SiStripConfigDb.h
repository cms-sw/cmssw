// Last commit: $Id: SiStripConfigDb.h,v 1.64 2008/04/30 13:32:13 bainbrid Exp $

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
  friend class SiStripPartition;
  friend class testSiStripConfigDb;


  // ---------- Typedefs ----------


  // FED connections
#ifdef USING_NEW_DATABASE_MODEL
  typedef ConnectionDescription FedConnection;
#else
  typedef FedChannelConnectionDescription FedConnection;
#endif
  typedef edm::MapOfVectors<std::string,FedConnection*> FedConnections;
  typedef FedConnections::range FedConnectionsRange;
  typedef std::vector<FedConnection*> FedConnectionV;
  
  // Device descriptions
  typedef enumDeviceType DeviceType;
  typedef deviceDescription DeviceDescription;
  typedef edm::MapOfVectors<std::string,DeviceDescription*> DeviceDescriptions;
  typedef DeviceDescriptions::range DeviceDescriptionsRange;
  typedef std::vector<DeviceDescription*> DeviceDescriptionV;
  
  // FED descriptions
  typedef Fed9U::Fed9UDescription FedDescription;
  typedef edm::MapOfVectors<std::string,FedDescription*> FedDescriptions;
  typedef FedDescriptions::range FedDescriptionsRange;
  typedef std::vector<FedDescription*> FedDescriptionV;

  // FED ids
  typedef std::vector<uint16_t> FedIds;
  typedef boost::iterator_range<FedIds::const_iterator> FedIdsRange;
  
  // DCU-DetId map
  typedef std::pair<uint32_t,TkDcuInfo*> DcuDetId; 
  typedef edm::MapOfVectors<std::string,DcuDetId> DcuDetIds; 
  typedef DcuDetIds::range DcuDetIdsRange; 
  typedef std::vector<DcuDetId> DcuDetIdV;
  typedef Sgi::hash_map<unsigned long,TkDcuInfo*> DcuDetIdMap;
  
  
  // Analysis descriptions
#ifdef USING_NEW_DATABASE_MODEL
  typedef CommissioningAnalysisDescription::commissioningType AnalysisType;
#else
  class CommissioningAnalysisDescription;
#endif
  typedef CommissioningAnalysisDescription AnalysisDescription;
  typedef edm::MapOfVectors<std::string,AnalysisDescription*> AnalysisDescriptions;
  typedef AnalysisDescriptions::range AnalysisDescriptionsRange;
  typedef std::vector<AnalysisDescription*> AnalysisDescriptionV;


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
  inline bool usingDb() const;
  
  /** Returns pointer to DeviceFactory API, with check if NULL. */
  DeviceFactory* const deviceFactory( std::string method_name = "" ) const;
  
  /** Returns pointer to DeviceFactory API, with check if NULL. */
  DbClient* const databaseCache( std::string method_name = "" ) const;
  
  
  // ---------- FED connections ----------


  /** Returns local cache (just for given partition if specified). */
  FedConnectionsRange getFedConnections( std::string partition = "" );

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
  DeviceDescriptionsRange getDeviceDescriptions( std::string partition = "" ); 

  /** Returns (pair of iterators to) descriptions of given type. */
  /** (APV25, APVMUX, DCU, LASERDRIVER, PLL). */
  //DeviceDescriptionsRange getDeviceDescriptions( DeviceType, std::string partition = "" );
  DeviceDescriptionsRange getDeviceDescriptions( DeviceType, std::string partition = "" );
  
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


  /** Returns local cache (just for given partition if specified). */
  FedDescriptionsRange getFedDescriptions( std::string partition = "" ); 

  /** Adds to local cache (just for given partition if specified). */
  void addFedDescriptions( std::string partition, std::vector<FedDescription*>& );
  
  /** Uploads to database (just for given partition if specified). */
  void uploadFedDescriptions( std::string partition = "" ); 
  
  /** Clears local cache (just for given partition if specified). */
  void clearFedDescriptions( std::string partition = "" ); 
  
  /** Prints local cache (just for given partition if specified). */
  void printFedDescriptions( std::string partition = "" ); 
  
  /** Extracts FED ids from FED descriptions. */
  FedIdsRange getFedIds( std::string partition = "" );
  
  /** Strip-level info enabled/disabled within FED descriptions. */
  inline bool usingStrips() const;
  
  /** Enables/disables strip-level info within FED descriptions. */
  inline void usingStrips( bool );
  

  // ---------- DCU-DetId info ----------


  /** Returns local cache (just for given partition if specified). */
  DcuDetIdsRange getDcuDetIds( std::string partition = "" );
  
  /** Adds to local cache (just for given partition if specified). */
  void addDcuDetIds( std::string partition, std::vector<DcuDetId>& );
  
  /** Uploads to database (just for given partition if specified). */
  void uploadDcuDetIds( std::string partition = "" );
  
  /** Clears local cache (just for given partition if specified). */
  void clearDcuDetIds( std::string partition = "" );
  
  /** Prints local cache (just for given partition if specified). */
  void printDcuDetIds( std::string partition = "" );
  
  /** Utility method. */ 
  static DcuDetIdV::const_iterator findDcuDetId( DcuDetIdV::const_iterator begin, 
						 DcuDetIdV::const_iterator end, 
						 uint32_t dcu_id  );
  
  /** Utility method. */ 
  static DcuDetIdV::iterator findDcuDetId( DcuDetIdV::iterator begin, 
					   DcuDetIdV::iterator end, 
					   uint32_t dcu_id  );
  

  // ---------- Commissioning analyses ---------- 

  
#ifdef USING_NEW_DATABASE_MODEL
  
  /** Returns local cache (just for given partition if specified). */
  AnalysisDescriptionsRange getAnalysisDescriptions( AnalysisType, std::string partition = "" );
  
  /** Adds to local cache (just for given partition if specified). */
  void addAnalysisDescriptions( std::string partition, std::vector<AnalysisDescription*>& );
  
  /** Uploads to database (just for given partition if specified). */
  void uploadAnalysisDescriptions( bool calibration_for_physics = false, std::string partition = "" );
  
  /** Clears local cache (just for given partition if specified). */
  void clearAnalysisDescriptions( std::string partition = "" );
  
  /** Prints local cache (just for given partition if specified). */
  void printAnalysisDescriptions( std::string partition = "" );
  
  /** Extracts unique hardware address of device from description. */
  DeviceAddress deviceAddress( const AnalysisDescription& ); //@@ uses temp offsets!
  
  /** Returns string for given analysis type. */
  std::string analysisType( AnalysisType ) const;
  
#endif
  

 private:


  // ---------- Private methods ----------


  /** */
  void clearLocalCache();

  /** */
  void usingDatabase();

  /** */
  void usingDatabaseCache();
  
  /** */
  void usingXmlFiles();
  
  /** Handles exceptions thrown by software. */
  void handleException( const std::string& method_name,
			const std::string& extra_info = "" ) const;
  
  /** Checks whether file at "path" exists or not. */
  bool checkFileExists( const std::string& path );
  
  /** Returns device identifier based on device type. */
  std::string deviceType( const enumDeviceType& device_type ) const;
  
  void clone( const DcuDetIdMap& in, DcuDetIdV& out ) const;
  
  void clone( const DcuDetIdV& in, DcuDetIdMap& out ) const;
  
  void clone( const DcuDetIdV& in, DcuDetIdV& out ) const;
  
  
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

  /** Fed9U descriptions. */
  FedDescriptions feds_;
 
  /** DcuId-DetId map (map of TkDcuInfo objects). */
  DcuDetIds dcuDetIds_;
  
#ifdef USING_NEW_DATABASE_MODEL

  /** Analysis descriptions for given commissioning run. */
  AnalysisDescriptions analyses_;

#endif

  /** Cache for devices of given type. */
  std::vector<DeviceDescription*> typedDevices_;
  
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
bool SiStripConfigDb::usingDb() const { return dbParams_.usingDb(); }

/** Indicates whether FED strip info is uploaded/downloaded. */
bool SiStripConfigDb::usingStrips() const { return usingStrips_; }

/** Switches on/off of upload/download for FED strip info. */
void SiStripConfigDb::usingStrips( bool using_strips ) { usingStrips_ = using_strips; }


#endif // OnlineDB_SiStripConfigDb_SiStripConfigDb_h

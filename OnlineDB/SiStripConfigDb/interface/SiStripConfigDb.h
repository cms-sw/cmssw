
#ifndef OnlineDB_SiStripConfigDb_SiStripConfigDb_h
#define OnlineDB_SiStripConfigDb_SiStripConfigDb_h

#define DATABASE  // Needed by DeviceFactory API! Do not comment!
//#define USING_DATABASE_MASKING

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "DataFormats/Common/interface/MapOfVectors.h"
#include "DataFormats/SiStripCommon/interface/ConstantsForRunType.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripDbParams.h"
#include "DeviceFactory.h"
#include "boost/range/iterator_range.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <ostream>
#include <vector>
#include <string>
#include <list>
#include <map>
#include <atomic>

#include "DbClient.h"
#include <cstdint>

namespace sistrip {
  static const uint16_t FEC_CRATE_OFFSET = 0;  //@@ temporary
  static const uint16_t FEC_RING_OFFSET = 0;   //@@ temporary
}  // namespace sistrip

// Friend class
namespace cms {
  class SiStripO2O;
}

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
  SiStripConfigDb(const edm::ParameterSet&, const edm::ActivityRegistry&);

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
  friend class SiStripCondObjBuilderFromDb;
  friend class SiStripPsuDetIdMap;

  // Commissioning clients
  friend class SiStripCommissioningDbClient;
  friend class SiStripCommissioningOfflineDbClient;
  friend class CommissioningHistosUsingDb;
  friend class FastFedCablingHistosUsingDb;
  friend class FedCablingHistosUsingDb;
  friend class ApvTimingHistosUsingDb;
  friend class OptoScanHistosUsingDb;
  friend class PedestalsHistosUsingDb;
  friend class PedsFullNoiseHistosUsingDb;
  friend class PedsOnlyHistosUsingDb;
  friend class NoiseHistosUsingDb;
  friend class VpspScanHistosUsingDb;
  friend class LatencyHistosUsingDb;
  friend class FineDelayHistosUsingDb;
  friend class CalibrationHistosUsingDb;
  friend class DaqScopeModeHistosUsingDb;

  // Utility and tests
  friend class SiStripPartition;
  friend class testSiStripConfigDb;

  // ---------- Typedefs ----------

  // FED connections
  typedef ConnectionDescription FedConnection;
  typedef edm::MapOfVectors<std::string, FedConnection*> FedConnections;
  typedef FedConnections::range FedConnectionsRange;
  typedef std::vector<FedConnection*> FedConnectionsV;

  // Device descriptions
  typedef enumDeviceType DeviceType;
  typedef deviceDescription DeviceDescription;
  typedef edm::MapOfVectors<std::string, DeviceDescription*> DeviceDescriptions;
  typedef DeviceDescriptions::range DeviceDescriptionsRange;
  typedef std::vector<DeviceDescription*> DeviceDescriptionsV;

  // FED descriptions
  typedef Fed9U::Fed9UDescription FedDescription;
  typedef edm::MapOfVectors<std::string, FedDescription*> FedDescriptions;
  typedef FedDescriptions::range FedDescriptionsRange;
  typedef std::vector<FedDescription*> FedDescriptionsV;

  // FED ids
  typedef std::vector<uint16_t> FedIds;
  typedef boost::iterator_range<FedIds::const_iterator> FedIdsRange;

  // DCU-DetId map
  typedef Sgi::hash_map<unsigned long, TkDcuInfo*> DcuDetIdMap;
  typedef std::pair<uint32_t, TkDcuInfo*> DcuDetId;
  typedef edm::MapOfVectors<std::string, DcuDetId> DcuDetIds;
  typedef DcuDetIds::range DcuDetIdsRange;
  typedef std::vector<DcuDetId> DcuDetIdsV;

  // Analysis descriptions
  typedef CommissioningAnalysisDescription::commissioningType AnalysisType;
  typedef CommissioningAnalysisDescription AnalysisDescription;
  typedef edm::MapOfVectors<std::string, AnalysisDescription*> AnalysisDescriptions;
  typedef AnalysisDescriptions::range AnalysisDescriptionsRange;
  typedef std::vector<AnalysisDescription*> AnalysisDescriptionsV;

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
  DeviceFactory* const deviceFactory(std::string method_name = "") const;

  /** Returns pointer to DeviceFactory API, with check if NULL. */
  DbClient* const databaseCache(std::string method_name = "") const;

  // ---------- Run numbers for partitions and run types ----------

  class Run {
  public:
    sistrip::RunType type_;
    std::string partition_;
    uint16_t number_;
    Run() : type_(sistrip::UNDEFINED_RUN_TYPE), partition_(""), number_(0) { ; }
  };

  typedef std::vector<Run> Runs;

  typedef std::map<sistrip::RunType, Runs> RunsByType;

  typedef std::map<std::string, Runs> RunsByPartition;

  /** Retrieves all runs from database. */
  void runs(Runs&) const;

  /** Runs organsed by type, optionally for given partition. */
  void runs(const Runs& in, RunsByType& out, std::string optional_partition = "") const;

  /** Runs organsed by partition, optionally for given type. */
  void runs(const Runs& in, RunsByPartition& out, sistrip::RunType optional_type = sistrip::UNDEFINED_RUN_TYPE) const;

  /** Retrieves all partitions names from database. */
  void partitions(std::list<std::string>&) const;

  //@@ SiStripPartition::Versions ???

  // ---------- FED connections ----------

  /** Returns local cache (just for given partition if specified). */
  FedConnectionsRange getFedConnections(std::string partition = "");

  /** Add to local cache (just for given partition if specified). */
  void addFedConnections(std::string partition, FedConnectionsV&);

  /** Uploads to database (just for given partition if specified). */
  void uploadFedConnections(std::string partition = "");

  /** Clears local cache (just for given partition if specified). */
  void clearFedConnections(std::string partition = "");

  /** Prints local cache (just for given partition if specified). */
  void printFedConnections(std::string partition = "");

  // ---------- FEC / Front-End devices ----------

  /** Returns local cache (just for given partition if specified). */
  DeviceDescriptionsRange getDeviceDescriptions(std::string partition = "");

  /** Returns (pair of iterators to) descriptions of given type. */
  /** (APV25, APVMUX, DCU, LASERDRIVER, PLL, DOH). */
  DeviceDescriptionsRange getDeviceDescriptions(DeviceType, std::string partition = "");

  /** Adds to local cache (just for given partition if specified). */
  void addDeviceDescriptions(std::string partition, DeviceDescriptionsV&);

  /** Uploads to database (just for given partition if specified). */
  void uploadDeviceDescriptions(std::string partition = "");

  /** Clears local cache (just for given partition if specified). */
  void clearDeviceDescriptions(std::string partition = "");

  /** Prints local cache (just for given partition if specified). */
  void printDeviceDescriptions(std::string partition = "");

  /** Extracts unique hardware address of device from description. */
  DeviceAddress deviceAddress(const deviceDescription&);  //@@ uses temp offsets!

  // ---------- FED descriptions ----------

  /** Returns local cache (just for given partition if specified). */
  FedDescriptionsRange getFedDescriptions(std::string partition = "");

  /** Adds to local cache (just for given partition if specified). */
  void addFedDescriptions(std::string partition, FedDescriptionsV&);

  /** Uploads to database (just for given partition if specified). */
  void uploadFedDescriptions(std::string partition = "");

  /** Clears local cache (just for given partition if specified). */
  void clearFedDescriptions(std::string partition = "");

  /** Prints local cache (just for given partition if specified). */
  void printFedDescriptions(std::string partition = "");

  /** Extracts FED ids from FED descriptions. */
  FedIdsRange getFedIds(std::string partition = "");

  /** Strip-level info enabled/disabled within FED descriptions. */
  inline bool usingStrips() const;

  /** Enables/disables strip-level info within FED descriptions. */
  inline void usingStrips(bool);

  // ---------- DCU-DetId info ----------

  /** Returns local cache (just for given partition if specified). */
  DcuDetIdsRange getDcuDetIds(std::string partition = "");

  /** Adds to local cache (just for given partition if specified). */
  void addDcuDetIds(std::string partition, DcuDetIdsV&);

  /** Uploads to database (just for given partition if specified). */
  void uploadDcuDetIds(std::string partition = "");

  /** Clears local cache (just for given partition if specified). */
  void clearDcuDetIds(std::string partition = "");

  /** Prints local cache (just for given partition if specified). */
  void printDcuDetIds(std::string partition = "");

  /** Utility method. */
  static DcuDetIdsV::const_iterator findDcuDetId(DcuDetIdsV::const_iterator begin,
                                                 DcuDetIdsV::const_iterator end,
                                                 uint32_t dcu_id);

  /** Utility method. */
  static DcuDetIdsV::iterator findDcuDetId(DcuDetIdsV::iterator begin, DcuDetIdsV::iterator end, uint32_t dcu_id);

  // ---------- Commissioning analyses ----------

  /** Returns local cache (just for given partition if specified). */
  AnalysisDescriptionsRange getAnalysisDescriptions(AnalysisType, std::string partition = "");

  /** Adds to local cache (just for given partition if specified). */
  void addAnalysisDescriptions(std::string partition, AnalysisDescriptionsV&);

  /** Uploads to database (just for given partition if specified). */
  void uploadAnalysisDescriptions(bool calibration_for_physics = false, std::string partition = "");

  /** Clears local cache (just for given partition if specified). */
  void clearAnalysisDescriptions(std::string partition = "");

  /** Prints local cache (just for given partition if specified). */
  void printAnalysisDescriptions(std::string partition = "");

  /** Extracts unique hardware address of device from description. */
  DeviceAddress deviceAddress(const AnalysisDescription&);  //@@ uses temp offsets!

  /** Returns string for given analysis type. */
  std::string analysisType(AnalysisType) const;

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
  void handleException(const std::string& method_name, const std::string& extra_info = "") const;

  /** Checks whether file at "path" exists or not. */
  bool checkFileExists(const std::string& path);

  /** Returns device identifier based on device type. */
  std::string deviceType(const enumDeviceType& device_type) const;

  void clone(const DcuDetIdMap& in, DcuDetIdsV& out) const;

  void clone(const DcuDetIdsV& in, DcuDetIdMap& out) const;

  void clone(const DcuDetIdsV& in, DcuDetIdsV& out) const;

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

  /** Analysis descriptions for given commissioning run. */
  AnalysisDescriptions analyses_;

  /** Cache for devices of given type. */
  DeviceDescriptionsV apvDevices_;

  /** Cache for devices of given type. */
  DeviceDescriptionsV muxDevices_;

  /** Cache for devices of given type. */
  DeviceDescriptionsV dcuDevices_;

  /** Cache for devices of given type. */
  DeviceDescriptionsV lldDevices_;

  /** Cache for devices of given type. */
  DeviceDescriptionsV pllDevices_;

  /** Cache for devices of given type. */
  DeviceDescriptionsV dohDevices_;

  /** Cache for devices of given type. */
  DeviceDescriptionsV typedDevices_;

  /** FED ids. */
  FedIds fedIds_;

  // ---------- Miscellaneous ----------

  /** Switch to enable/disable transfer of strip information. */
  bool usingStrips_;

  /** */
  bool openConnection_;

  /** Static counter of instances of this class. */
  static std::atomic<uint32_t> cntr_;

  static std::atomic<bool> allowCalibUpload_;
};

// ---------- Inline methods ----------

/** Returns database connection parameters. */
const SiStripDbParams& SiStripConfigDb::dbParams() const { return dbParams_; }

/** Indicates whether DB (true) or XML files (false) are used. */
bool SiStripConfigDb::usingDb() const { return dbParams_.usingDb(); }

/** Indicates whether FED strip info is uploaded/downloaded. */
bool SiStripConfigDb::usingStrips() const { return usingStrips_; }

/** Switches on/off of upload/download for FED strip info. */
void SiStripConfigDb::usingStrips(bool using_strips) { usingStrips_ = using_strips; }

#endif  // OnlineDB_SiStripConfigDb_SiStripConfigDb_h

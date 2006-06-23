// Last commit: $Id: SiStripConfigDb.h,v 1.6 2006/06/21 13:30:14 bainbrid Exp $
// Latest tag:  $Name:  $
// Location:    $Source: /cvs_server/repositories/CMSSW/CMSSW/OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h,v $

#ifndef SiStripConfigDb_H
#define SiStripConfigDb_H

#define DATABASE //@@ necessary?

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DeviceFactory.h"
#include "boost/cstdint.hpp"
#include <vector>
#include <string>

class SiStripFedCabling;
class SiStripFecCabling;

/**	
   \class SiStripConfigDb
   \brief An interface class to the DeviceFactory
   \author R.Bainbridge
*/
class SiStripConfigDb {
  
 public:
  
  // -------------------- Constructors, destructors --------------------
  
  /** Constructor when using the configuration database, which takes
      as arguments the database connection parameters. */
  SiStripConfigDb( std::string db_user, 
		   std::string db_passwd, 
		   std::string db_path,
		   std::string db_partition ); 

  /** Constructor when using xml files, which takes as arguments the
      paths to the various input (and output) xml files. */
  SiStripConfigDb( std::string input_module_xml,
		   std::string input_dcuinfo_xml,
		   std::vector<std::string> input_fec_xmls,
		   std::vector<std::string> input_fed_xmls,
		   std::string output_module_xml = "/tmp/module.xml",
		   std::string output_dcuinfo_xml = "/tmp/dcuinfo.xml",
		   std::vector<std::string> output_fec_xmls = std::vector<std::string>(1,"/tmp/fec.xml"),
		   std::vector<std::string> output_fed_xmls = std::vector<std::string>(1,"/tmp/fed.xml") );
  
  /** Default destructor. */
  ~SiStripConfigDb();
  
  // -------------------- Typedefs, structs and enums --------------------

  typedef std::vector<deviceDescription*>               DeviceDescriptions;
  typedef std::vector<Fed9U::Fed9UDescription*>         FedDescriptions;
  typedef std::vector<FedChannelConnectionDescription*> FedConnections;
  typedef Sgi::hash_map<unsigned long,TkDcuInfo*>       DcuDetIdMap;
  typedef std::vector<piaResetDescription*>             PiaResetDescriptions;
  typedef std::vector<TkDcuConversionFactors*>          DcuConversions;
  
  /** Struct containing partition name and version. */
  struct Partition { 
    std::string name_;
    uint32_t major_;
    uint32_t minor_;
  };
  
  /** Struct that holds addresses that uniquely identify a hardware
      component within the control system. */
  struct DeviceAddress { 
    uint16_t fecCrate_; 
    uint16_t fecSlot_;
    uint16_t fecRing_;
    uint16_t ccuAddr_;
    uint16_t ccuChan_;
    uint16_t i2cAddr_;
  };

  // -------------------- Connection and local cache --------------------
  
  /** Establishes connection to DeviceFactory API. */
  void openDbConnection();
  
  /** Closes connection to DeviceFactory API. */
  void closeDbConnection();
  
  /** Returns whether using database or xml files. */
  inline const bool& usingDb() const;
  
  /** Returns pointer to DeviceFactory API, with check if NULL. */
  DeviceFactory* const deviceFactory( std::string method_name = "" ) const;
  
  /** Resets and clears all local caches and synchronizes with
      descriptions retrieved from database or xml files. */
  void refreshLocalCaches();
  
  // -------------------- Partitioning and versioning --------------------
  
  /** Returns name and major/minor versions for current partition. */
  inline const Partition& getPartitionNameAndVersion() const;

  /** Sets partition name and version. */
  inline void setPartitionNameAndVersion( const Partition& );

  /** Sets partition name and version. */
  inline void setPartitionNameAndVersion( const std::string& partition_name,
					  const uint32_t& major_version,
					  const uint32_t& minor_version );

  // ---------------------------------------------------------------------------
  // ---------- "Get" methods (to retrieve descriptions) -----------------------
  // ---------------------------------------------------------------------------
  
  /** Returns descriptions for a given device type (which can be one
      of the following: APV25, APVMUX, DCU, LASERDRIVER, PLL ). If
      boolean is set to true, the descriptions of all devices EXCEPT
      those of the given type are returned. */
  const DeviceDescriptions& getDeviceDescriptions( const enumDeviceType&,
						   bool all_devices_except = false ); 
  
  /** Fills local cache with all device descriptions from DB/xml. */
  const DeviceDescriptions& getDeviceDescriptions(); 
  
  /** Fills local cache with FED descriptions from DB/xml. */
  const FedDescriptions& getFedDescriptions();

  /** Fills local cache with connection descriptions from DB/xml. */
  const FedConnections& getFedConnections();

  /** Fills local cache with PIA reset descriptions from DB/xml. */
  const PiaResetDescriptions& getPiaResetDescriptions();

  // -------------------- FEC / Front-End devices -------------------- 
  
  
  /** Overwrites local cache of device descriptions. */
  void setDeviceDescriptions( const DeviceDescriptions& ) {;} 
  
  /** Resets and clears local cache. */
  void resetDeviceDescriptions(); 
  
  /** Uploads all device descriptions to DB/xml. */
  void uploadDeviceDescriptions( bool new_major_version = false ); 
  
  /** Extracts unique hardware address of device from description. */
  const DeviceAddress& deviceAddress( const deviceDescription& );
  
  // -------------------- FED descriptions --------------------

  
  /** Overwrites local cache of FED descriptions. */
  void setFedDescriptions( const FedDescriptions& );

  /** Resets and clears local cache. */
  void resetFedDescriptions();
  
  /** Uploads FED descriptions to DB/xml. */
  void uploadFedDescriptions( bool new_major_version = false );
  
  /** Create "dummy" FED descriptions based on FED cabling. */
  void createFedDescriptions( const SiStripFedCabling&,
			      FedDescriptions& );
  
  /** Extracts FED ids from FED descriptions. */
  const std::vector<uint16_t>& getFedIds();
  
  /** Indicates if strip info is enabled/disabled within FED descs. */
  inline const bool& usingStrips() const;
  
  /** Enable/disable strip info within FED descriptions. */
  inline void usingStrips( bool );
  
  // -------------------- FED connections --------------------
  
  /** Overwrites local cache of FED-FEC connections. */
  void setFedConnections( const FedConnections& ) {;} //@@ to be implemented!
  
  /** Resets and clears local cache. */
  void resetFedConnections();
  
  /** Uploads FED-FEC connections to DB/xml. */
  void uploadFedConnections( bool new_major_version = false );
  
  /** Creates "dummy" FED connections based on FED cabling. */
  void createFedConnections( const SiStripFedCabling&,
			     FedConnections& );
  
  // -------------------- DCU info --------------------

  /** Returns the DcuId-DetId map. If the local cache is empty, it
      retrieves the DcuId-DetId map from the DB/xml file. */
  const DcuDetIdMap& getDcuDetIdMap();
  
  /** Clears the local cache of DcuId-DetId map and sets it equal to
      the map provided within the argument list. */
  void setDcuDetIdMap( const DcuDetIdMap& ) {;}
  
  /** Resets and clears local cache. */
  void resetDcuDetIdMap();
  
  /** Uploads the contents of the local cache to DB/xml file. */
  void uploadDcuDetIdMap();

  
  // -------------------- PIA reset descriptions --------------------

  
  /** Overwrites local cache of PIA reset descriptions. */
  void setPiaResetDescriptions( const PiaResetDescriptions& ) {;}

  /** Resets and clears local cache. */
  void resetPiaResetDescriptions();
  
  /** Uploads PIA reset descriptions to DB/xml. */
  void uploadPiaResetDescriptions();
  
  /** Create "dummy" PIA reset descriptions based on FED cabling. */
  void createPiaResetDescriptions( const SiStripFedCabling&,
				   PiaResetDescriptions& ) {;}
  
  // -------------------- DCU conversion factors --------------------
  
  /*   void setDcuConversionFactors ( tkDcuInfoVector vDcuInfoPartition,  */
  /* 				 DeviceFactory &deviceFactory,  */
  /* 				 std::string partitionName ) throw (FecExceptionHandler); */

  // -------------------- Miscellaneous --------------------
  
  /** Creates "dummy" descriptions based on FEC cabling. */
  void createDescriptions( const SiStripFecCabling&,
			   DeviceDescriptions&,
			   PiaResetDescriptions pia_resets,
			   DcuConversions dcu_convs );
  
 private:

  // -------------------- Misc private methods --------------------
  
  /** */
  void usingDatabase();
  
  /** */
  void usingXmlFiles();
  
  /** Handles exceptions thrown by FEC and FED software. */
  void handleException( const std::string& method_name,
			const std::string& extra_info = "" ) throw (cms::Exception);
  
  /** Checks whether file at "path" exists or not. */
  bool checkFileExists( const std::string& path );
  
  /** Returns device identifier based on device type. */
  std::string deviceType( const enumDeviceType& device_type ) const;
  
  // ---------- Database connection, partitions and versions ----------

  /** Pointer to the DeviceFactory API. */
  DeviceFactory* factory_; 

  /** Configuration database connection parameter: "user name". */
  std::string user_;

  /** Configuration database connection parameter: "password". */
  std::string passwd_;

  /** Configuration database connection parameter: "path". */
  std::string path_;

  /** Partition name and version. */
  Partition partition_;
  
  // -------------------- Xml file input/output --------------------
  
  /** Switch to identify whether using configuration database or not
      (if not, then the xml files are used). */
  bool usingDb_;

  /** Path to input "module.xml" file containing hardware connections. */
  std::string inputModuleXml_;

  /** Path to input "DcuInfo" xml file that contains DcuId-DetId map and
      other parameters from static table. */
  std::string inputDcuInfoXml_;

  /** Paths to input FEC xml file(s) containing device information. */
  std::vector<std::string> inputFecXml_;

  /** Paths to input FED "description" xml file(s). */
  std::vector<std::string> inputFedXml_;

  /** Path to output "module.xml" file containing hardware connections. */
  std::string outputModuleXml_;

  /** Path to output "DcuInfo" xml file that contains DcuId-DetId map and
      other parameters from static table. */
  std::string outputDcuInfoXml_;

  /** Paths to output FEC xml file(s) containing device information. */
  std::vector<std::string> outputFecXml_;

  /** Paths to output FED "description" xml file(s). */
  std::vector<std::string> outputFedXml_;

  // -------------------- Local cache --------------------

  /** Vector of descriptions for all FEC devices (including DCUs). */
  DeviceDescriptions devices_;

  /** PIA reset descriptions are necessary when using xml files. */
  std::vector<piaResetDescription*> piaResets_;

  /** Fed9U descriptions. */
  FedDescriptions feds_;

  /** FED-FEC connection descriptions. */
  FedConnections connections_;

  /** TkDcuInfo objects, containing DcuId-DetId map. */
  DcuDetIdMap dcuDetIdMap_;

  /** Indicates local cache of device descriptions has been reset. */
  bool resetDevices_;

  /** Indicates local cache of PIA descriptions has been reset. */
  bool resetPiaResets_;

  /** Indicates local cache of FED descriptions has been reset. */
  bool resetFeds_;

  /** Indicates local cache of FED connections has been reset. */
  bool resetConnections_;

  /** Indicates local cache of Dcu-DetId map has been reset. */
  bool resetDcuDetIdMap_;

  // -------------------- Miscellaneous --------------------
  
  /** Switch to enable/disable transfer of strip information. */
  bool usingStrips_;

};

// -------------------- Inline methods --------------------

const bool& SiStripConfigDb::usingDb() const { return usingDb_; }

const SiStripConfigDb::Partition& SiStripConfigDb::getPartitionNameAndVersion() const { return partition_; }
void SiStripConfigDb::setPartitionNameAndVersion( const SiStripConfigDb::Partition& partition ) { partition_ = partition; }
void SiStripConfigDb::setPartitionNameAndVersion( const std::string& partition_name,
						  const uint32_t& major_version,
						  const uint32_t& minor_version ) { 
  partition_.name_ = partition_name;
  partition_.major_ = major_version;
  partition_.minor_ = minor_version;
}

void SiStripConfigDb::usingStrips( bool using_strips ) { usingStrips_ = using_strips; }
const bool& SiStripConfigDb::usingStrips() const { return usingStrips_; }

#endif // SiStripConfigDb_H




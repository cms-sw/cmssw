// Last commit: $Id$
// Latest tag:  $Name$
// Location:    $Source$

#ifndef SiStripConfigDb_H
#define SiStripConfigDb_H

#define DATABASE //@@ necessary?

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
		   std::string db_path ); 

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
  
  // -------------------- Typedefs and structs --------------------

  /** */
  typedef std::vector<deviceDescription*> DeviceDescriptions;
  /** */
  typedef std::vector<Fed9U::Fed9UDescription*> FedDescriptions;
  /** */
  typedef std::vector<FedChannelConnectionDescription*> FedConnections;
  /** */
  typedef Sgi::hash_map<unsigned long,TkDcuInfo*> DcuIdDetIdMap;
  /** */
  typedef std::vector<piaResetDescription*> PiaResetDescriptions;
  /** */
  typedef std::vector<TkDcuConversionFactors*> DcuConversions;
  
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
  
  // ---------- Database connection, partitions and versions ----------
  
  /** */
  inline DeviceFactory* const deviceFactory() const;

  /** Establishes connection to DeviceFactory API. */
  bool openDbConnection();

  /** Closes connection to DeviceFactory API. */
  bool closeDbConnection();

  /** Returns whether using database or xml files. */
  inline const bool& usingDb() const;
  
  /** Returns name and major/minor versions for current partition. */
  inline const Partition& getPartitionNameAndVersion() const;

  /** Sets partition name and version. */
  inline void setPartitionNameAndVersion( const Partition& );

  /** Sets partition name and version. */
  inline void setPartitionNameAndVersion( const std::string& partition_name,
					  const uint32_t& major_version,
					  const uint32_t& minor_version );
  
  // -------------------- Front-end devices -------------------- 
  
  /** Fills local cache with device descriptions from DB/xml. */
  const DeviceDescriptions& getDeviceDescriptions(); 
  
  /** Overwrites local cache of device descriptions. */
  void setDeviceDescriptions( const DeviceDescriptions& ); 
  
  /** Uploads device descriptions to DB/xml. */
  void uploadDeviceDescriptions( bool new_major_version = false ); 
  
  /** Creates "dummy" device descriptions based on FEC cabling. */
  void createDeviceDescriptions( const SiStripFecCabling&,
				 DeviceDescriptions& );
  
  /** Returns descriptions for a given device type (which can be one
      of the following: APV25, APVMUX, DCU, LASERDRIVER, PLL ). */
  const DeviceDescriptions& getDeviceDescriptions( const enumDeviceType& ); 
  
  /** Extracts unique hardware address of device from description. */
  const DeviceAddress& deviceAddress( const deviceDescription& );
  
  // -------------------- Front-end driver --------------------

  /** Fills local cache with FED descriptions from DB/xml. */
  const FedDescriptions& getFedDescriptions();
  
  /** Overwrites local cache of FED descriptions. */
  void setFedDescriptions( const FedDescriptions& );
  
  /** Uploads FED descriptions to DB/xml. */
  void uploadFedDescriptions( bool new_major_version = false );
  
  /** Create "dummy" FED descriptions based on FED cabling. */
  void createFedDescriptions( const SiStripFedCabling&,
			      FedDescriptions& );
  
  /** Extracts FED ids from FED descriptions. */
  const std::vector<uint16_t>& getFedIds();
  
  /** Enable/disable strip info within FED descriptions. */
  inline void usingStrips( bool );
  
  // -------------------- Fed connections --------------------
  
  /** Fills local cache with connection descriptions from DB/xml. */
  const FedConnections& getFedConnections();
  
  /** Overwrites local cache of FED-FEC connections. */
  void setFedConnections( const FedConnections& );
  
  /** Uploads FED-FEC connections to DB/xml. */
  void uploadFedConnections( bool new_major_version = false );
  
  /** Creates "dummy" FED connections based on FED cabling. */
  void createFedConnections( const SiStripFedCabling&,
			     FedConnections& );
  
  /** Identifies whether the cached device descriptions are used to
      generate the FEC cabling (either because the FED connections are
      not available in the DB/xml or the user has inhibited it). */
  inline bool buildFecCablingFromFecDevices();
  
  /** Switch that determines whether the cached device descriptions
      are used to generate the FEC cabling. */
  inline void buildFecCablingFromFecDevices( const bool& );
  
  // -------------------- DCU info --------------------

  /** Returns the DcuId-DetId map. If the local cache is empty, it
      retrieves the DcuId-DetId map from the DB/xml file. */
  const DcuIdDetIdMap& getDcuIdDetIdMap();
  
  /** Clears the local cache of DcuId-DetId map and sets it equal to
      the map provided within the argument list. */
  void setDcuIdDetIdMap( const DcuIdDetIdMap& ) {;}
  
  /** Uploads the contents of the local cache to DB/xml file. */
  void uploadDcuIdDetIdMap();
  
  /** Clears the local cache. */
  void clearDcuIdDetIdMap() {;}

  // -------------------- DCU conversion factors --------------------
  
/*   void setDcuConversionFactors ( tkDcuInfoVector vDcuInfoPartition,  */
/* 				 DeviceFactory &deviceFactory,  */
/* 				 std::string partitionName ) throw (FecExceptionHandler); */
  
 private:

  // -------------------- Misc private methods --------------------
  
  /** */
  bool xmlFileIO();

  /** Method for handling exceptions thrown by FEC software. */
  void handleFecException( const std::string& method_name );

  /** Method for handling exceptions thrown by FED software. */
  void handleFedException( const std::string& method_name );

  /** Method for handling exceptions thrown by Oracle. */
  void handleSqlException( const std::string& method_name );

  /** Checks whether file at "path" exists or not. */
  bool checkFileExists( const std::string& path );

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

  /** Vector of descriptions for all FEC devices (except DCU). */
  DeviceDescriptions devices_;

  /** Vector of descriptions for all DCU devices. */
  DeviceDescriptions dcus_;

  /** PIA reset descriptions are necessary when using xml files. */
  std::vector<piaResetDescription*> piaResets_;

  /** Fed9U descriptions. */
  FedDescriptions feds_;

  /** FED-FEC connection descriptions. */
  FedConnections connections_;

  /** TkDcuInfo objects, containing DcuId-DetId map. */
  DcuIdDetIdMap dcuIdDetIdMap_;

  // -------------------- Miscellaneous --------------------
  
  /** Switch to enable/disable transfer of strip information. */
  bool usingStrips_;

  /** */
  bool useDeviceDescriptions_;

};

// -------------------- Inline methods --------------------

DeviceFactory* const SiStripConfigDb::deviceFactory() const { return factory_; }

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

bool SiStripConfigDb::buildFecCablingFromFecDevices() { return (inputModuleXml_=="") || useDeviceDescriptions_; }
void SiStripConfigDb::buildFecCablingFromFecDevices( const bool& use_devices ) { useDeviceDescriptions_ = use_devices; }

#endif // SiStripConfigDb_H




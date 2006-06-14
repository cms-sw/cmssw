#ifndef SiStripConfigDb_H
#define SiStripConfigDb_H

#define DATABASE //@@ <- this is needed if using database???

// database
#include "DeviceFactory.h"
#include "Fed9UDeviceFactoryLib.hh"
#include "DbAccess.h"
// connections
#include "FedPmcXmlDescription.h"
// fec
#include "deviceDescription.h"
#include "apvDescription.h"
#include "muxDescription.h"
#include "pllDescription.h"
#include "dcuDescription.h"
#include "laserdriverDescription.h"
//#include "TkDcuInfo.h"
//#include "TkDcuConversionFactors.h"
#include "deviceType.h"
#include "tscTypes.h"
#include "keyType.h"
#include "FecExceptionHandler.h"
// fed
#include "Fed9UUtils.hh"
// boost
#include "boost/cstdint.hpp"
// std
#include <vector>
#include <string>

/**	
   \class SiStripConfigDb
   \brief An interface class to the DeviceFactory
   \author R.Bainbridge
   \version 0.1
   \date 17/01/06
*/
class SiStripConfigDb {

 public: // ----- PUBLIC INTERFACE -----
  
  SiStripConfigDb( std::string user, 
		   std::string passwd, 
		   std::string path, 
		   std::string partition = "" ); 
  ~SiStripConfigDb();
  
  // ----- TYPEDEFS AND STRUCTS -----

  struct DeviceAddress { 
    int16_t fecCrate; 
    int16_t fecSlot;
    int16_t fecRing;
    int16_t ccuAddr;
    int16_t ccuChan;
    int16_t i2cAddr;
  };

  void fromXml( const bool& from_xml ) { fromXml_ = from_xml; }
  void xmlFile( const std::string& xml_file ) { xmlFile_ = xml_file; }

  // ----- DATABASE AND PARTITIONS -----

  /** Returns pointer to DeviceFactory object. */
  inline DeviceFactory* deviceFactory() { return factory_; }

  /** Returns partition name. */
  std::string partitionName();
  /** Returns major/minor versions for given partition name. */
  pair<int16_t,int16_t> partitionVersion( std::string partition_name );
  
  // ----- FRONT-END <-> FED CONNECTIONS -----
  
  std::vector<FedChannelConnectionDescription*>& fedConnections( bool clear_cache = false );
  
  // ----- FRONT END DEVICES ----- 
  
  /** Returns HW addresses uniquely identifying a device. */
  DeviceAddress hwAddresses( deviceDescription& description );
  /** Returns all devices correponding to a given device type. */
  void feDevices( enumDeviceType device_type, deviceVector& devices );
  /** Returns APV descriptions. */
  void apvDescriptions( std::vector<apvDescription*>& apv_descriptions );
  /** Returns DCU descriptions. */
  void dcuDescriptions( std::vector<dcuDescription*>& dcu_descriptions );
  /** Returns laser driver (AOH) descriptions. */
  void aohDescriptions( std::vector<laserdriverDescription*>& aoh_descriptions ) {;}
  
  // ----- FRONT END DRIVER -----
  
  /** Returns FED descriptions. */
  void fedDescriptions( std::vector<Fed9U::Fed9UDescription*>& fed_descriptions );
  /** Returns FED identifiers. */
  void fedIds( std::vector<uint16_t>& fed_ids );
  
  // ----- DCUs ------

/*   /\** *\/ */
/*   void setDefaultDcuConversionFactors( const std::vector<TkDcuInfo*>&,  */
/* 				       const std::string& partition_name ) throw (FecExceptionHandler); */

 private: // ----- PRIVATE METHODS -----
  
  bool openDbConnection();
  bool closeDbConnection();
  void deviceSummary();
  
 private: // ----- PRIVATE DATA MEMBERS -----

  /** Private constructor. */
  SiStripConfigDb() {;}

  // ----- DATABASE-RELATED -----
  
  DeviceFactory* factory_; 
  std::string user_;
  std::string passwd_;
  std::string path_;
  bool fromXml_;
  std::string xmlFile_;

  // ----- PARTITIONS AND VERSIONING -----

  std::string partition_;
  
  // ----- DEVICES AND DESCRIPTIONS -----

  deviceVector allDevices_;
  deviceVector apvDevices_; //@@ needed?
  deviceVector dcuDevices_; //@@ needed?
  std::vector<FedChannelConnectionDescription*> fedConnections_;

};

#endif // SiStripConfigDb_H




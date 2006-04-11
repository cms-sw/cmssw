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
#include "dcuDescription.h"
#include "laserdriverDescription.h"
#include "tscTypes.h"
#include "keyType.h"
// fed
#include "Fed9UUtils.hh"
// boost
#include "boost/cstdint.hpp"
// std
#include <vector>
#include <string>

using namespace std;

/**	
   \class SiStripConfigDb
   \brief An interface class to the DeviceFactory
   \author R.Bainbridge
   \version 0.1
   \date 17/01/06
*/
class SiStripConfigDb {

 public: // ----- PUBLIC INTERFACE -----
  
  SiStripConfigDb( string user, 
		   string passwd, 
		   string path, 
		   string partition = "" ); 
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
  void xmlFile( const string& xml_file ) { xmlFile_ = xml_file; }

  // ----- PARTITION HANDLING -----

  /** Returns partition name. */
  string partitionName();
  /** Returns major/minor versions for given partition name. */
  pair<int16_t,int16_t> partitionVersion( string partition_name );
  
  // ----- FRONT-END <-> FED CONNECTIONS -----
  
  vector<FedChannelConnectionDescription*>& fedConnections( bool clear_cache = false );
  
  // ----- FRONT END CONTROLLER AND FE DEVICES ----- 
  
  /** Returns HW addresses uniquely identifying a device. */
  DeviceAddress hwAddresses( deviceDescription& description );
  /** Returns all devices correponding to a given device type. */
  void feDevices( enumDeviceType device_type, deviceVector& devices );
  /** Returns APV descriptions. */
  void apvDescriptions( vector<apvDescription*>& apv_descriptions );
  /** Returns DCU descriptions. */
  void dcuDescriptions( vector<dcuDescription*>& dcu_descriptions );
  /** Returns laser driver (AOH) descriptions. */
  void aohDescriptions( vector<laserdriverDescription*>& aoh_descriptions ) {;}
  
  // ----- FRONT END DRIVER -----
  
  /** Returns FED descriptions. */
  void fedDescriptions( vector<Fed9U::Fed9UDescription*>& fed_descriptions );
  /** Returns FED identifiers. */
  void fedIds( vector<uint16_t>& fed_ids );
  
 private: // ----- PRIVATE METHODS -----
  
  bool openDbConnection();
  bool closeDbConnection();
  void deviceSummary();
  
 private: // ----- PRIVATE DATA MEMBERS -----

  /** Private constructor. */
  SiStripConfigDb() {;}

  // ----- DATABASE-RELATED -----
  
  DeviceFactory* factory_; 
  string user_;
  string passwd_;
  string path_;
  bool fromXml_;
  string xmlFile_;

  // ----- PARTITIONS AND VERSIONING -----

  string partition_;
  
  // ----- DEVICES AND DESCRIPTIONS -----

  deviceVector allDevices_;
  deviceVector apvDevices_; //@@ needed?
  deviceVector dcuDevices_; //@@ needed?
  vector<FedChannelConnectionDescription*> fedConnections_;

};

#endif // SiStripConfigDb_H




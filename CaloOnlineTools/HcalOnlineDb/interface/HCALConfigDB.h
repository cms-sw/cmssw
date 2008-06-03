//
// Gena Kukartsev (Brown), Feb. 1, 2008
//
//
#ifndef HCALConfigDB_h
#define HCALConfigDB_h

#include <iostream>
#include <string.h>
#include "CaloOnlineTools/HcalOnlineDb/interface/ConfigurationDatabaseImpl.hh"

using namespace std;
using namespace hcal;

/**

   \class HCALConfigDB
   \brief Gather config data from online DB
   \author Gena Kukartsev

*/

class HCALConfigDB{
 public:
  
  HCALConfigDB( );
  ~HCALConfigDB( );
  HCALConfigDB( string _accessor );
  void connect( string _accessor );
  void connect( string _accessor1, string _accessor2 ); // for very specific case of XML and Oracle
  void disconnect( void );
  void setAccessor( string _accessor );
  std::vector<unsigned int> getOnlineLUT( string tag, int crate, int slot, int topbottom, int fiber, int channel, int luttype );
  std::vector<unsigned int> getOnlineLUT( string tag, uint32_t _rawid, hcal::ConfigurationDatabase::LUTType _lt = hcal::ConfigurationDatabase::LinearizerLUT );
  std::vector<unsigned int> getOnlineLUTFromXML( string tag, uint32_t _rawid, hcal::ConfigurationDatabase::LUTType _lt = hcal::ConfigurationDatabase::LinearizerLUT );

  oracle::occi::Connection * getConnection( void );
  oracle::occi::Environment * getEnvironment( void );

  std::string clobToString(oracle::occi::Clob);
  
 protected:

  string accessor;
  ConfigurationDatabaseImpl * database;

  // the second DB handle for a crazy case
  // when two connections are needed,
  // e.g. when the main connection is to
  // an XML file
  string accessor2;
  ConfigurationDatabaseImpl * database2;

};
#endif

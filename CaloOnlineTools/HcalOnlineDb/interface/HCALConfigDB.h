//
// Gena Kukartsev (Brown), Feb. 1, 2008
//
//
#ifndef HCALConfigDB_h
#define HCALConfigDB_h

#include <iostream>
#include <string.h>
#include "CaloOnlineTools/HcalOnlineDb/interface/ConfigurationDatabaseImpl.hh"


/**

   \class HCALConfigDB
   \brief Gather config data from online DB
   \author Gena Kukartsev

*/

class HCALConfigDB{
 public:
  typedef hcal::ConfigurationDatabaseImpl ConfigurationDatabaseImpl;
  
  HCALConfigDB( );
  ~HCALConfigDB( );
  HCALConfigDB( std::string _accessor );
  void connect( std::string _accessor );
  void connect( std::string _accessor1, std::string _accessor2 ); // for very specific case of XML and Oracle
  void disconnect( void );
  void setAccessor( std::string _accessor );
  std::vector<unsigned int> getOnlineLUT( std::string tag, int crate, int slot, int topbottom, int fiber, int channel, int luttype );
  std::vector<unsigned int> getOnlineLUT( std::string tag, uint32_t _rawid, hcal::ConfigurationDatabase::LUTType _lt = hcal::ConfigurationDatabase::LinearizerLUT );
  std::vector<unsigned int> getOnlineLUTFromXML( std::string tag, uint32_t _rawid, hcal::ConfigurationDatabase::LUTType _lt = hcal::ConfigurationDatabase::LinearizerLUT );

  oracle::occi::Connection * getConnection( void );
  oracle::occi::Environment * getEnvironment( void );

  std::string clobToString(const oracle::occi::Clob&);
  
 protected:

  std::string accessor;
  ConfigurationDatabaseImpl * database;

  // the second DB handle for a crazy case
  // when two connections are needed,
  // e.g. when the main connection is to
  // an XML file
  std::string accessor2;
  ConfigurationDatabaseImpl * database2;

};
#endif

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
  
  HCALConfigDB( ){ };
  HCALConfigDB( string _accessor );
  void connect( string _accessor );
  void disconnect( void );
  void setAccessor( string _accessor );
  std::vector<unsigned int> getOnlineLUT( string tag, int crate, int slot, int topbottom, int fiber, int channel, int luttype );
  std::vector<unsigned int> getOnlineLUT( string tag, uint32_t _rawid, hcal::ConfigurationDatabase::LUTType _lt = hcal::ConfigurationDatabase::LinearizerLUT );
  //hcal::ConfigurationDatabase::LUTId getLUTId( uint32_t _rawid );
  
 protected:

  string accessor;
  ConfigurationDatabaseImpl * database;

};
#endif

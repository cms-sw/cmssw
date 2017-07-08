//
// Gena Kukartsev (Brown), Feb. 22, 2008
//
//
#ifndef HcalQIEManager_h
#define HcalQIEManager_h

#include <iostream>
#include <string.h>
#include "CaloOnlineTools/HcalOnlineDb/interface/HCALConfigDB.h"

/**

   \class HcalQIEManager
   \brief Various manipulations with QIE and QIE ADC
   \author Gena Kukartsev

*/

class HcalChannelId{

 public:
  
  HcalChannelId(){};
  ~HcalChannelId(){};
  
  int eta, phi, depth;
  std::string subdetector;

  bool operator<( const HcalChannelId & other) const;
    
};

class HcalQIECaps{

 public:

  HcalQIECaps(){};
  ~HcalQIECaps(){};

  // cap0 offset x 4, cap1 offset x 4...
  // cap0 slope  x 4, cap1 slope  x 4...
  double caps[32];
};

class HcalQIEManager{
 public:
  
  HcalQIEManager( );
  ~HcalQIEManager( );
  
  std::map<HcalChannelId,HcalQIECaps> & getQIETableFromFile( const std::string& _filename );
  void getTableFromDb( const std::string& query_file, const std::string& output_file );
  int generateQieTable( const std::string& db_file, const std::string& old_file, const std::string& output_file );
  int getHfQieTable( const std::string& input_file, const std::string& output_file );

  static std::vector <std::string> splitString (const std::string& fLine);

 protected:


};
#endif

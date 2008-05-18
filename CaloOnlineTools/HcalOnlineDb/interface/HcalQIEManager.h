//
// Gena Kukartsev (Brown), Feb. 22, 2008
//
//
#ifndef HcalQIEManager_h
#define HcalQIEManager_h

#include <iostream>
#include <string.h>
#include "CaloOnlineTools/HcalOnlineDb/interface/HCALConfigDB.h"

using namespace std;
using namespace hcal;

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
  
  std::map<HcalChannelId,HcalQIECaps> & getQIETableFromFile( std::string _filename );
  void getTableFromDb( string query_file, string output_file );
  int generateQieTable( string db_file, string old_file, string output_file );
  int getHfQieTable( string input_file, string output_file );

  static std::vector <std::string> splitString (const std::string& fLine);

 protected:


};
#endif

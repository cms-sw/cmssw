#ifndef HcalLutManager_h
#define HcalLutManager_h

#include <iostream>
#include <string>
#include <vector>
#include "CaloOnlineTools/HcalOnlineDb/interface/LutXml.h"

/**

   \class HcalLutManager
   \brief Various manipulations with trigger Lookup Tables
   \author Gena Kukartsev, Brown University, March 14, 2008

*/

class XMLDOMBlock;

class HcalLutManager{
 public:
  
  HcalLutManager( );
  ~HcalLutManager( );

  void init( void );
  std::string & getLutXml( std::vector<unsigned int> & _lut );

 protected:
  
  LutXml * lut_xml;

};
#endif

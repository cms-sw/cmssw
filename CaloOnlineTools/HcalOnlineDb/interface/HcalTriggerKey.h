#ifndef HCALConfigDBTools_XMLTools_HcalTriggerKey_h
#define HCALConfigDBTools_XMLTools_HcalTriggerKey_h
// -*- C++ -*-
//
// Package:     XMLTools
// Class  :     HcalTriggerKey
// 
/**\class HcalTriggerKey HcalTriggerKey.h CaloOnlineTools/HcalOnlineDb/interface/HcalTriggerKey.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Gena Kukartsev, kukarzev@fnal.gov
//         Created:  Tue Nov 06 14:30:33 CDT 2007
//

#include <map>

#include "CalibCalorimetry/HcalTPGAlgos/interface/XMLDOMBlock.h"
#include "CalibCalorimetry/HcalTPGAlgos/interface/XMLProcessor.h"

class HcalTriggerKey : public XMLDOMBlock
{
  
 public:
  
  HcalTriggerKey();
  virtual ~HcalTriggerKey();

  int init(void);
  int add_data( std::string id, std::string type, std::string value);
  // fills the XML document with the key. Returns number of configs in the key
  int fill_key( std::string key_id, std::map<std::string, std::string> & key);  
  int compose_key_dialogue( void );
  
 protected:

  XERCES_CPP_NAMESPACE::MemBufInputSource * _root; // a container for the XML template;
  XERCES_CPP_NAMESPACE::MemBufInputSource * _data; // a container for the XML template;

 private:
  HcalTriggerKey(const HcalTriggerKey&); // stop default
  const HcalTriggerKey& operator=(const HcalTriggerKey&); // stop default

  
};

#endif

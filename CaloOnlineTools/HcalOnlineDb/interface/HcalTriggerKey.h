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
// $Id: HcalTriggerKey.h,v 1.2 2008/09/01 17:14:23 kukartse Exp $
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
  int add_data( string id, string type, string value);
  // fills the XML document with the key. Returns number of configs in the key
  int fill_key( string key_id, std::map<string, string> & key);  
  int compose_key_dialogue( void );
  
 protected:

  MemBufInputSource * _root; // a container for the XML template;
  MemBufInputSource * _data; // a container for the XML template;

 private:
  HcalTriggerKey(const HcalTriggerKey&); // stop default
  const HcalTriggerKey& operator=(const HcalTriggerKey&); // stop default

  
};

#endif

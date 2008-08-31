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
// $Id: HcalTriggerKey.h,v 1.2 2008/03/07 02:49:13 kukartse Exp $
//

// system include files

// user include files
#include "CaloOnlineTools/HcalOnlineDb/interface/XMLDOMBlock.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/XMLProcessor.h"

// forward declarations

class HcalTriggerKey : public XMLDOMBlock
{
  
 public:
  
  HcalTriggerKey();
  virtual ~HcalTriggerKey();

  int init(void);
  int add_data( string id, string type, string value);
  
 protected:

  MemBufInputSource * _root; // a container for the XML template;
  MemBufInputSource * _data; // a container for the XML template;

 private:
  HcalTriggerKey(const HcalTriggerKey&); // stop default
  const HcalTriggerKey& operator=(const HcalTriggerKey&); // stop default

  
};

#endif

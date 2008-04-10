#ifndef HCALConfigDBTools_XMLTools_HcalHardwareXml_h
#define HCALConfigDBTools_XMLTools_HcalHardwareXml_h
// -*- C++ -*-
//
// Package:     XMLTools
// Class  :     HcalHardwareXml
// Original Author:  Gena Kukartsev, kukarzev@fnal.gov
// 
/**\class HcalHardwareXml HcalHardwareXml.h CaloOnlineTools/HcalOnlineDb/interface/HcalHardwareXml.h

 Description: <one line class summary>

 Usage:
    <usage>

*/

#include <map>

#include "CaloOnlineTools/HcalOnlineDb/interface/XMLDOMBlock.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/XMLProcessor.h"

using namespace std;

typedef struct _HcalPart
  {
    _HcalPart(){ mode=""; kind_of_part=""; name_label=""; barcode=""; comment=""; attr_name=""; attr_value=""; };
    string mode;
    string kind_of_part;
    string name_label;
    string barcode;
    string comment;
    string attr_name;
    string attr_value;
  } HcalPart;



class HcalHardwareXml : public XMLDOMBlock
{
  
 public:
    
  HcalHardwareXml();
  HcalHardwareXml( std::string _type );
  virtual ~HcalHardwareXml();
  
  int addHardware( std::map<string,map<string,map<string,map<int,string> > > > & hw_map );

 private:

  HcalHardwareXml(const HcalHardwareXml&); // stop default
  const HcalHardwareXml& operator=(const HcalHardwareXml&); // stop default

  DOMElement * addPart( DOMElement * parent, HcalPart & part );

  DOMElement * partsElem;

  //hw_map["rbx_slot"]["rbx"]["rm"][qie_slot]="qie";
  //std::map<string,map<string,map<string,map<int,string> > > > hw_map;
  
};


#endif

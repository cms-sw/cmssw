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

#include "CalibCalorimetry/HcalTPGAlgos/interface/XMLDOMBlock.h"
#include "CalibCalorimetry/HcalTPGAlgos/interface/XMLProcessor.h"

typedef struct _HcalPart
  {
    _HcalPart(){ mode=""; kind_of_part=""; name_label=""; barcode=""; comment=""; attr_name=""; attr_value=""; };
    std::string mode;
    std::string kind_of_part;
    std::string name_label;
    std::string barcode;
    std::string comment;
    std::string attr_name;
    std::string attr_value;
  } HcalPart;



class HcalHardwareXml : public XMLDOMBlock
{
  
 public:
    
  HcalHardwareXml();
  HcalHardwareXml( std::string _type );
  virtual ~HcalHardwareXml();
  
  int addHardware( std::map<std::string,std::map<std::string,std::map<std::string,std::map<int,std::string> > > > & hw_map );

 private:

  HcalHardwareXml(const HcalHardwareXml&); // stop default
  const HcalHardwareXml& operator=(const HcalHardwareXml&); // stop default

  XERCES_CPP_NAMESPACE::DOMElement * addPart( XERCES_CPP_NAMESPACE::DOMElement * parent, HcalPart & part );

  XERCES_CPP_NAMESPACE::DOMElement * partsElem;

  //hw_map["rbx_slot"]["rbx"]["rm"][qie_slot]="qie";
  //std::map<std::string,std::map<std::string,std::map<std::string,std::map<int,std::string> > > > hw_map;
  
};


#endif

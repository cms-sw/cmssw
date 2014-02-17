#ifndef CaloOnlineTools_HcalOnlineDb_HcalL1TriggerObjectsXml_h
#define CaloOnlineTools_HcalOnlineDb_HcalL1TriggerObjectsXml_h
// -*- C++ -*-
//
// Package:     CaloOnlineTools/HcalOnlineDb
// Class  :     HcalL1TriggerObjectsXml
// 
/**\class HcalL1TriggerObjectsXml HcalL1TriggerObjectsXml.h CaloOnlineTools/HcalOnlineDb/interface/HcalL1TriggerObjectsXml.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Gena Kukartsev, kukarzev@fnal.gov
//         Created:  Wed Jul 01 06:42:00 CDT 2009
// $Id: HcalL1TriggerObjectsXml.h,v 1.1 2009/09/23 22:06:28 kukartse Exp $
//

#include <map>
#include <string>
#include "CaloOnlineTools/HcalOnlineDb/interface/HcalChannelDataXml.h"

class HcalL1TriggerObjectsXml : public HcalChannelDataXml
{
  
 public:

  HcalL1TriggerObjectsXml(const std::string& input_tag_name);
  virtual ~HcalL1TriggerObjectsXml();
  
  // add dataset to the XML document
  DOMNode * add_hcal_channel_dataset( int ieta, int iphi, int depth, std::string subdetector,
                                      double ped, double gain, int flag);
  
  // add data tag inside a dataset tag
  DOMElement * add_data( DOMNode * _dataset, double ped, double gain, int flag);
};

#endif

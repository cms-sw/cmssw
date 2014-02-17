// -*- C++ -*-
//
// Package:     CaloOnlineTools/HcalOnlineDb
// Class  :     HcalL1TriggerObjectsXml
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Gena Kukartsev, kukarzev@fnal.gov
//         Created:  Wed Jul 01 06:30:00 CDT 2009
// $Id: HcalL1TriggerObjectsXml.cc,v 1.2 2009/11/18 23:02:26 lsexton Exp $
//

#include <iostream>
#include <string>
#include <vector>

#include "CaloOnlineTools/HcalOnlineDb/interface/HcalL1TriggerObjectsXml.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/ConnectionManager.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/ConfigurationDatabaseException.hh"
#include "xgi/Utils.h"
#include "toolbox/string.h"
#include "OnlineDB/Oracle/interface/Oracle.h"

using namespace std;
using namespace oracle::occi;

HcalL1TriggerObjectsXml::HcalL1TriggerObjectsXml(const std::string& input_tag_name)
{
  extension_table_name="HCAL_L1_TRIGGER_OBJECTS_V1";
  type_name="HCAL L1 Trigger Objects [V1]";
  run_number = 1;
  channel_map = "HCAL_CHANNELS";
  data_set_id = -1;
  iov_id = 1;
  iov_begin = 1;
  iov_end = -1;
  tag_id = 2;
  tag_mode = "auto";
  //tag_name = "test_L1TriggerObjects";
  tag_name = input_tag_name;
  detector_name = "HCAL";
  comment = hcal_ass.getRandomQuote();
  tag_idref = 2;
  iov_idref = 1;
  data_set_idref = -1;
  init_data();
}


HcalL1TriggerObjectsXml::~HcalL1TriggerObjectsXml()
{
}


DOMElement * HcalL1TriggerObjectsXml::add_data( DOMNode * _dataset, double ped, double gain, int flag){
  DOMElement * _data_elem = get_data_element(_dataset);
  add_element(_data_elem, XMLProcessor::_toXMLCh("AVERAGE_PEDESTAL"), XMLProcessor::_toXMLCh(ped));
  add_element(_data_elem, XMLProcessor::_toXMLCh("RESPONSE_CORRECTED_GAIN"), XMLProcessor::_toXMLCh(gain));
  add_element(_data_elem, XMLProcessor::_toXMLCh("FLAG"), XMLProcessor::_toXMLCh(flag));
  //
  return _data_elem;
}


DOMNode * HcalL1TriggerObjectsXml::add_hcal_channel_dataset( int ieta, int iphi, int depth, std::string subdetector,
                                                           double ped, double gain, int flag){
  DOMNode * _dataset = add_dataset();
  add_hcal_channel(_dataset, ieta, iphi, depth, subdetector);
  add_data(_dataset, ped, gain, flag);
  return _dataset;
}

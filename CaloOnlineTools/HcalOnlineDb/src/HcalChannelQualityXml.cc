// -*- C++ -*-
//
// Package:     CaloOnlineTools/HcalOnlineDb
// Class  :     HcalChannelQualityXml
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Gena Kukartsev, kukarzev@fnal.gov
//         Created:  Wed Jul 01 06:30:00 CDT 2009
// $Id: HcalChannelQualityXml.cc,v 1.2 2009/06/24 09:49:11 kukartse Exp $
//

#include <iostream>
#include <string>
#include <vector>

#include "CaloOnlineTools/HcalOnlineDb/interface/HcalChannelQualityXml.h"

using namespace std;



HcalChannelQualityXml::HcalChannelQualityXml()
/*
  :extension_table_name(""),
  type_name(""),
  run_number(-1),
  channel_map("HCAL_CHANNELS"),
  data_set_id(-1),
  iov_id(1),
  iov_begin(1),
  iov_end(-1),
  tag_id(2),
  tag_mode("auto"),
  tag_name("test_tag_name"),
  detector_name("HCAL"),
  comment(""),
  tag_idref(2),
  iov_idref(1),
  data_set_idref(-1)
*/
{
  extension_table_name="HCAL_CHANNEL_QUALITY_V1";
  type_name="HCAL Channel Quality [V1]";
  run_number = -1;
  channel_map = "HCAL_CHANNELS";
  data_set_id = -1;
  iov_id = 1;
  iov_begin = 1;
  iov_end = -1;
  tag_id = 2;
  tag_mode = "auto";
  tag_name = "test_channel_quality_tag_name";
  detector_name = "HCAL";
  comment = "";
  tag_idref = 2;
  iov_idref = 1;
  data_set_idref = -1;
  //
  init_data();
}


HcalChannelQualityXml::~HcalChannelQualityXml()
{
}


DOMElement * HcalChannelQualityXml::add_data( DOMNode * _dataset, int _channel_status, int _on_off, std::string _comment ){
  DOMElement * _data_elem = get_data_element(_dataset);
  add_element(_data_elem, XMLProcessor::_toXMLCh("CHANNEL_STATUS_WORD"), XMLProcessor::_toXMLCh(_channel_status));
  add_element(_data_elem, XMLProcessor::_toXMLCh("CHANNEL_ON_OFF_STATE"), XMLProcessor::_toXMLCh(_on_off));
  add_element(_data_elem, XMLProcessor::_toXMLCh("COMMENT_DESCRIPTION"), XMLProcessor::_toXMLCh(_comment));
  //
  return _data_elem;
}


DOMNode * HcalChannelQualityXml::add_hcal_channel_dataset( int ieta, int iphi, int depth, std::string subdetector,
							   int _channel_status, int _on_off, std::string _comment ){
  DOMNode * _dataset = add_dataset();
  add_hcal_channel(_dataset, ieta, iphi, depth, subdetector);
  add_data(_dataset, _channel_status, _on_off, _comment);
  return _dataset;
}

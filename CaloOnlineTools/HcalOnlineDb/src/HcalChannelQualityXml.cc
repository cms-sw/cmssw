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
// $Id: HcalChannelQualityXml.cc,v 1.1 2009/07/09 17:00:13 kukartse Exp $
//

#include <iostream>
#include <string>
#include <vector>

#include "CaloOnlineTools/HcalOnlineDb/interface/HcalChannelQualityXml.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/HcalChannelIterator.h"

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
  comment = hcal_ass.getRandomQuote();
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


int HcalChannelQualityXml::set_all_channels_on_off( int _hb, int _he, int _hf, int _ho){
  HcalChannelIterator iter;
  iter.initHBEFListFromLmapAscii();
  std::string _subdetector = "";
  int _onoff = -1;
  std::string _comment = get_random_comment();
  for (iter.begin(); !iter.end(); iter.next()){
    HcalSubdetector _det = iter.getHcalSubdetector();
    if (_det == HcalBarrel){
      _subdetector = "HB";
      _onoff = _hb;
    }
    else if (_det == HcalEndcap){
      _subdetector = "HE";
      _onoff = _he;
    }
    if (_det == HcalForward){
      _subdetector = "HF";
      _onoff = _hf;
    }
    if (_det == HcalOuter){
      _subdetector = "HO";
      _onoff = _ho;
    }
    add_hcal_channel_dataset( iter.getIeta(), iter.getIphi(), iter.getDepth(), _subdetector,
			      0, _onoff, _comment );
    
  }

  return 0;
}



std::string HcalChannelQualityXml::get_random_comment( void ){
  return hcal_ass.getRandomQuote();
}

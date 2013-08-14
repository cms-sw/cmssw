// -*- C++ -*-
//
// Package:     CaloOnlineTools/HcalOnlineDb
// Class  :     HcalChannelDataXml
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Gena Kukartsev, kukarzev@fnal.gov
//         Created:  Wed Jul 01 06:30:00 CDT 2009
// $Id: HcalChannelDataXml.cc,v 1.5 2009/10/26 02:55:16 kukartse Exp $
//

#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <iconv.h>
#include <sys/time.h>
#include "CaloOnlineTools/HcalOnlineDb/interface/HcalChannelDataXml.h"

using namespace std;




HcalChannelDataXml::HcalChannelDataXml(){
  comment = hcal_ass.getRandomQuote();
  //init_data();
  dataset_count = 0;
  global_timestamp = time(NULL);
}


HcalChannelDataXml::~HcalChannelDataXml()
{
  delete _root;
  delete _dataset;
}


int HcalChannelDataXml::init_data( void )
{
  static const char * _str = "\
<ROOT>\n\
  <HEADER>\n\
    <TYPE>\n\
      <EXTENSION_TABLE_NAME>HCAL_CHANNEL_QUALITY_V1</EXTENSION_TABLE_NAME>\n\
      <NAME>HCAL Channel Quality [V1]</NAME>\n\
    </TYPE>\n\
    <RUN>\n\
      <RUN_NUMBER>1</RUN_NUMBER>\n\
    </RUN>\n\
    <HINTS channelmap='HCAL_CHANNELS'/>\n\
  </HEADER>\n\
  <ELEMENTS>\n\
    <DATA_SET id='-1'/>\n\
    <IOV id='1'>\n\
      <INTERVAL_OF_VALIDITY_BEGIN>1</INTERVAL_OF_VALIDITY_BEGIN>\n\
      <INTERVAL_OF_VALIDITY_END>-1</INTERVAL_OF_VALIDITY_END>\n\
    </IOV>\n\
    <TAG id='2' mode='auto'>\n\
      <TAG_NAME>test-channel-status-01jul2009-v4</TAG_NAME>\n\
      <DETECTOR_NAME>HCAL</DETECTOR_NAME>\n\
      <COMMENT_DESCRIPTION>testing channel status recording</COMMENT_DESCRIPTION>\n\
    </TAG>\n\
  </ELEMENTS>\n\
  <MAPS>\n\
    <TAG idref='2'>\n\
      <IOV idref='1'>\n\
        <DATA_SET idref='-1'/>\n\
      </IOV>\n\
    </TAG>\n\
  </MAPS>\n\
  </ROOT>\n\
  ";
  const XMLByte * _template = (const XMLByte *)_str;
  _root = new MemBufInputSource( _template, strlen( (const char *)_template ), "_root", false );
  parse(*_root);
  //
  //_____ fill header tags that will be redefined in the derived classes
  //
  set_header_table_name(extension_table_name);
  set_header_type(type_name);
  set_header_run_number(run_number);
  set_header_channel_map(channel_map);
  set_elements_dataset_id(data_set_id);
  set_elements_iov_id(iov_id);
  set_elements_iov_begin(iov_begin);
  set_elements_iov_end(iov_end);
  set_elements_tag_id(tag_id);
  set_elements_tag_mode(tag_mode);
  set_elements_tag_name(tag_name);
  set_elements_detector_name(detector_name);
  set_elements_comment(comment);
  set_maps_tag_idref(tag_idref);
  set_maps_iov_idref(iov_idref);
  set_maps_dataset_idref(data_set_idref);
  //
  static const char * _str2 =  "\
  <DATA_SET>\n\
    <COMMENT_DESCRIPTION>test-channel-status-26jun2009-v1</COMMENT_DESCRIPTION>\n\
    <CREATE_TIMESTAMP>2009-04-06 17:36:40.0</CREATE_TIMESTAMP>\n\
    <CREATED_BY_USER>user</CREATED_BY_USER>\n\
    <VERSION>test-channel-status-26jun2009-v1</VERSION>\n\
    <SUBVERSION>1</SUBVERSION>\n\
    <CHANNEL>\n\
      <EXTENSION_TABLE_NAME>HCAL_CHANNELS</EXTENSION_TABLE_NAME>\n\
    </CHANNEL>\n\
    <DATA>\n\
    </DATA>\n\
  </DATA_SET>\n\
  ";
  const XMLByte * _template2 = (const XMLByte *)_str2;
  _dataset = new MemBufInputSource( _template2, strlen( (const char *)_template2 ), "_dataset", false );

  //
  //_____ some more default initialization
  username = hcal_ass.getUserName();
  dataset_comment = hcal_ass.getRandomQuote();
  //
  return 0;
}


DOMNode * HcalChannelDataXml::add_hcal_channel( DOMNode * _dataset, int ieta, int iphi, int depth, std::string subdetector ){
  DOMElement * _channel = get_channel_element(_dataset);
  if (_channel){
    add_element(_channel, XMLProcessor::_toXMLCh("IETA"), XMLProcessor::_toXMLCh(ieta));
    add_element(_channel, XMLProcessor::_toXMLCh("IPHI"), XMLProcessor::_toXMLCh(iphi));
    add_element(_channel, XMLProcessor::_toXMLCh("DEPTH"), XMLProcessor::_toXMLCh(depth));
    add_element(_channel, XMLProcessor::_toXMLCh("SUBDET"), XMLProcessor::_toXMLCh(subdetector));
  }
  
  return (DOMNode *)_channel;
}


DOMNode * HcalChannelDataXml::add_dataset( void ){
  XMLDOMBlock dataset_block( *_dataset );
  DOMDocument * dataset_doc = dataset_block . getDocument();
  DOMElement * root_elem = (DOMElement *)(document -> getElementsByTagName( XMLProcessor::_toXMLCh( "ROOT" ) ) -> item(0));
  DOMNode * dataset_node = document -> importNode( dataset_doc -> getDocumentElement(), true );
  root_elem -> appendChild( dataset_node );
  //
  //_____ set defaults
  //
  //
  //_____ fix due to the new convention: version/subversion combo must be unique for every payload
  //
  char _buf[128];
  //time_t _offset = time(NULL);
  time_t _offset = global_timestamp;
  sprintf( _buf, "%d", (uint32_t)_offset );
  std::string _version;
  _version.clear();
  _version.append(tag_name);
  _version.append(".");
  _version.append(_buf);
  //
  DOMElement * dataset_elem = (DOMElement *)dataset_node;
  setTagValue(dataset_elem, "COMMENT_DESCRIPTION", dataset_comment);
  setTagValue(dataset_elem, "CREATE_TIMESTAMP", getTimestamp(time(0)));
  setTagValue(dataset_elem, "CREATED_BY_USER", username);
  setTagValue(dataset_elem, "VERSION", _version);
  setTagValue(dataset_elem, "SUBVERSION", dataset_count);
  //
  //_____ set the chanel table name consistent with the header HINT
  //
  setTagValue(dataset_elem, "EXTENSION_TABLE_NAME", channel_map);
  //
  if (dataset_node) ++dataset_count;
  return dataset_node;
}


DOMElement * HcalChannelDataXml::get_data_element( DOMNode * _dataset ){
  DOMElement * _element = 0;
  if (_dataset){
    _element = (DOMElement *)(((DOMElement *)_dataset) -> getElementsByTagName(XMLProcessor::_toXMLCh("DATA"))->item( 0 ));
  }
  return _element;
}


DOMElement * HcalChannelDataXml::get_channel_element( DOMNode * _dataset ){
  DOMElement * _element = 0;
  if (_dataset){
    _element = (DOMElement *)(((DOMElement *)_dataset) -> getElementsByTagName(XMLProcessor::_toXMLCh("CHANNEL"))->item( 0 ));
  }
  return _element;
}


DOMNode * HcalChannelDataXml::set_header_table_name(std::string name){
  extension_table_name = name;
  DOMElement * _header = (DOMElement *)(document -> getElementsByTagName( XMLString::transcode("HEADER") ) -> item(0));
  return setTagValue(_header, "EXTENSION_TABLE_NAME", extension_table_name);
}


DOMNode * HcalChannelDataXml::set_header_type(std::string type){
  type_name = type;
  DOMElement * _header = (DOMElement *)(document -> getElementsByTagName( XMLString::transcode("HEADER") ) -> item(0));
  return setTagValue(_header, "NAME", type_name);
}


DOMNode * HcalChannelDataXml::set_header_run_number(int run){
  run_number = run;
  DOMElement * _header = (DOMElement *)(document -> getElementsByTagName( XMLString::transcode("HEADER") ) -> item(0));
  return setTagValue(_header, "RUN_NUMBER", run_number);
}


DOMNode * HcalChannelDataXml::set_header_channel_map(std::string name){
  channel_map = name;
  DOMElement * _header = (DOMElement *)(document -> getElementsByTagName( XMLString::transcode("HEADER") ) -> item(0));
  return setTagAttribute(_header, "HINTS", "channelmap", channel_map);
}


DOMNode * HcalChannelDataXml::set_elements_dataset_id(int id){
  data_set_id = id;
  DOMElement * _elements = (DOMElement *)(document -> getElementsByTagName( XMLString::transcode("ELEMENTS") ) -> item(0));
  return setTagAttribute(_elements, "DATA_SET", "id", data_set_id);
}


DOMNode * HcalChannelDataXml::set_elements_iov_id(int id){
  iov_id = id;
  DOMElement * _elements = (DOMElement *)(document -> getElementsByTagName( XMLString::transcode("ELEMENTS") ) -> item(0));
  return setTagAttribute(_elements, "IOV", "id", iov_id);
}


DOMNode * HcalChannelDataXml::set_elements_iov_begin(int value){
  iov_begin = value;
  DOMElement * _elements = (DOMElement *)(document -> getElementsByTagName( XMLString::transcode("ELEMENTS") ) -> item(0));
  return setTagValue(_elements, "INTERVAL_OF_VALIDITY_BEGIN", iov_begin);
}


DOMNode * HcalChannelDataXml::set_elements_iov_end(int value){
  iov_end = value;
  DOMElement * _elements = (DOMElement *)(document -> getElementsByTagName( XMLString::transcode("ELEMENTS") ) -> item(0));
  return setTagValue(_elements, "INTERVAL_OF_VALIDITY_END", iov_end);
}


DOMNode * HcalChannelDataXml::set_elements_tag_id(int value){
  tag_id = value;
  DOMElement * _elements = (DOMElement *)(document -> getElementsByTagName( XMLString::transcode("ELEMENTS") ) -> item(0));
  return setTagAttribute(_elements, "TAG", "id", tag_id);
}


DOMNode * HcalChannelDataXml::set_elements_tag_mode(std::string value){
  tag_mode = value;
  DOMElement * _elements = (DOMElement *)(document -> getElementsByTagName( XMLString::transcode("ELEMENTS") ) -> item(0));
  return setTagAttribute(_elements, "TAG", "mode", tag_mode);
}


DOMNode * HcalChannelDataXml::set_elements_tag_name(std::string value){
  tag_name = value;
  DOMElement * _elements = (DOMElement *)(document -> getElementsByTagName( XMLString::transcode("ELEMENTS") ) -> item(0));
  return setTagValue(_elements, "TAG_NAME", tag_name);
}


DOMNode * HcalChannelDataXml::set_elements_detector_name(std::string value){
  detector_name = value;
  DOMElement * _elements = (DOMElement *)(document -> getElementsByTagName( XMLString::transcode("ELEMENTS") ) -> item(0));
  return setTagValue(_elements, "DETECTOR_NAME", detector_name);
}


DOMNode * HcalChannelDataXml::set_elements_comment(std::string value){
  comment = value;
  DOMElement * _elements = (DOMElement *)(document -> getElementsByTagName( XMLString::transcode("ELEMENTS") ) -> item(0));
  return setTagValue(_elements, "COMMENT_DESCRIPTION", comment);
}


DOMNode * HcalChannelDataXml::set_maps_tag_idref(int value){
  tag_idref = value;
  DOMElement * _maps = (DOMElement *)(document -> getElementsByTagName( XMLString::transcode("MAPS") ) -> item(0));
  return setTagAttribute(_maps, "TAG", "idref", tag_idref);
}


DOMNode * HcalChannelDataXml::set_maps_iov_idref(int value){
  iov_idref = value;
  DOMElement * _maps = (DOMElement *)(document -> getElementsByTagName( XMLString::transcode("MAPS") ) -> item(0));
  return setTagAttribute(_maps, "IOV", "idref", iov_idref);
}


DOMNode * HcalChannelDataXml::set_maps_dataset_idref(int value){
  data_set_idref = value;
  DOMElement * _maps = (DOMElement *)(document -> getElementsByTagName( XMLString::transcode("MAPS") ) -> item(0));
  return setTagAttribute(_maps, "DATA_SET", "idref", data_set_idref);
}









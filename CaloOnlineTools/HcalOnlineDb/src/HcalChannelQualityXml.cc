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
// $Id: HcalChannelQualityXml.cc,v 1.12 2010/08/06 20:24:12 wmtan Exp $
//

#include <iostream>
#include <string>
#include <vector>

#include "CaloOnlineTools/HcalOnlineDb/interface/HcalChannelQualityXml.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/HcalChannelIterator.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/ConnectionManager.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/ConfigurationDatabaseException.hh"
#include "xgi/Utils.h"
#include "toolbox/string.h"
#include "OnlineDB/Oracle/interface/Oracle.h"

using namespace std;
using namespace oracle::occi;


HcalChannelQualityXml::ChannelQuality::_ChannelQuality(){
  status = -1;
  onoff = -1;
  comment = "Comment has not been entered";
}


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
  geomid_cq.clear();
  detid_cq.clear();
  //
  //hAss.getListOfChannelsFromDb();
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


int HcalChannelQualityXml::set_all_channels_status( uint32_t _hb,
						    uint32_t _he,
						    uint32_t _hf,
						    uint32_t _ho){
  HcalChannelIterator iter;
  iter.initHBEFListFromLmapAscii();
  std::string _subdetector = "";
  uint32_t _status = 0;
  int _onoff = 0;
  std::string _comment = get_random_comment();
  for (iter.begin(); !iter.end(); iter.next()){
    HcalSubdetector _det = iter.getHcalSubdetector();
    if (_det == HcalBarrel){
      _subdetector = "HB";
      _status = _hb;
    }
    else if (_det == HcalEndcap){
      _subdetector = "HE";
      _status = _he;
    }
    else if (_det == HcalForward){
      _subdetector = "HF";
      _status = _hf;
    }
    else if (_det == HcalOuter){
      _subdetector = "HO";
      _status = _ho;
    }
    else continue;
    add_hcal_channel_dataset( iter.getIeta(), iter.getIphi(), iter.getDepth(), _subdetector,
			      _status, _onoff, _comment );
  }
  return 0;
}



std::string HcalChannelQualityXml::get_random_comment( void ){
  return hcal_ass.getRandomQuote();
}



//
//_____ reads on/off channel states from OMDS for a given tag and IOV
//      returns number of channels retrieved
int HcalChannelQualityXml::getBaseLineFromOmds(std::string _tag, int _iov_begin){
  static ConnectionManager conn;
  conn.connect();
  std::string query = "select ";
  //query            += "       sp.record_id as record_id ";
  query            += "       sp.subdet as subdetector ";
  query            += "       ,sp.ieta as IETA ";
  query            += "       ,sp.iphi as IPHI ";
  query            += "       ,sp.depth as DEPTH ";
  query            += "       ,sp.channel_status_word as STATUS_WORD ";
  query            += "       ,sp.channel_on_off_state as ON_OFF ";
  query            += "       ,sp.commentdescription ";
  //query            += "       ,sp.channel_map_id detid ";
  //query            += "       ,sp.interval_of_validity_begin as IOV_BEGIN ";
  //query            += "       ,sp.interval_of_validity_end as IOV_END ";
  query            += "from ";
  query            += "       ( ";
  query            += "       select MAX(cq.record_id) as record_id ";
  query            += "             ,MAX(cq.interval_of_validity_begin) as iov_begin ";
  query            += "             ,cq.channel_map_id ";
  query            += "       from ";
  query            += "             cms_hcl_hcal_cond.v_hcal_channel_quality cq ";
  query            += "       where ";
  query            += "             tag_name=:1 ";
  query            += "       and";
  query            += "       cq.interval_of_validity_begin<=:2";
  query            += "       group by ";
  query            += "             cq.channel_map_id ";
  query            += "       order by ";
  query            += "             cq.channel_map_id ";
  query            += "       ) fp ";
  query            += "inner join ";
  query            += "       cms_hcl_hcal_cond.v_hcal_channel_quality sp ";
  query            += "on ";
  query            += "       fp.record_id=sp.record_id ";
  int _n_channels = 0;
  try {
    oracle::occi::Statement* stmt = conn.getStatement(query);
    stmt->setString(1,_tag);
    stmt->setInt(2,_iov_begin);
    oracle::occi::ResultSet *rs = stmt->executeQuery();
    geomid_cq.clear();
    //detid_cq.clear();
    while (rs->next()) {
      _n_channels++;
      int _geomId   = hAss.getGeomId(hAss.getSubdetector(rs->getString(1)),
				rs->getInt(2),
				rs->getInt(3),
				rs->getInt(4)
				);
      HcalChannelQualityXml::ChannelQuality _cq;
      _cq.status  = rs->getInt(5);
      _cq.onoff   = rs->getInt(6);
      _cq.comment = rs->getString(7);
      geomid_cq.insert(std::pair<int, HcalChannelQualityXml::ChannelQuality>(_geomId, _cq));
    }
  }
  catch (SQLException& e) {
    std::cerr << ::toolbox::toString("Oracle  exception : %s",e.getMessage().c_str()) << std::endl;
    XCEPT_RAISE(hcal::exception::ConfigurationDatabaseException,::toolbox::toString("Oracle  exception : %s",e.getMessage().c_str()));
  }
  conn.disconnect();
  return _n_channels;
}


// map key is the proprietary channel hash, see HcalAssistant
int HcalChannelQualityXml::addChannelQualityGeom(std::map<int,ChannelQuality> & _cq){
  int _n_channels = 0;
  for (std::map<int,ChannelQuality>::const_iterator _chan=_cq.begin();
       _chan!=_cq.end();
       _chan++
       ){
    add_hcal_channel_dataset( hAss.getHcalIeta(_chan->first),
			      hAss.getHcalIphi(_chan->first),
			      hAss.getHcalDepth(_chan->first),
			      hAss.getSubdetectorString(hAss.getHcalSubdetector(_chan->first)),
			      _chan->second.status,
			      _chan->second.onoff,
			      _chan->second.comment
			      );
    _n_channels++;
  }
  return _n_channels;
}


int HcalChannelQualityXml::addChannelToGeomIdMap( int ieta, int iphi, int depth, std::string subdetector,
			   int _channel_status, int _on_off, std::string _comment ){
  int _geomId = hAss.getGeomId(hAss.getSubdetector(subdetector),
			       ieta,
			       iphi,
			       depth);
  HcalChannelQualityXml::ChannelQuality _cq;
  _cq.status  = _channel_status;
  _cq.onoff   = _on_off;
  _cq.comment = _comment;
  if (geomid_cq.find(_geomId)==geomid_cq.end()){
    geomid_cq.insert(std::pair<int, HcalChannelQualityXml::ChannelQuality>(_geomId,_cq));
  }
  else{
    geomid_cq[_geomId]=_cq;
  }
  return 0;
}


int HcalChannelQualityXml::readStatusWordFromStdin(std::string base){
  std::string _row;
  int _lines = 0;
  ChannelQuality _cq;
  _cq.onoff   = 0;
  _cq.status  = 0;
  _cq.comment = "filled from an ASCII stream";
  geomid_cq.clear();
  while ( getline( std::cin, _row ) > 0 ){
    //#(empty) eta phi dep det value DetId(optional)
    int _eta, _phi, _dep, _value;
    char _det[32];
    int _read;
    if ( base.find("hex")!=std::string::npos ){
      const char * _format = "%d %d %d %s %X";
      _read = sscanf( _row . c_str(), _format,
		      &_eta, &_phi, &_dep, _det, &_value
		      );
    }
    else if ( base.find("dec")!=std::string::npos ){
      const char * _format = "%d %d %d %s %d";
      _read = sscanf( _row . c_str(), _format,
		      &_eta, &_phi, &_dep, _det, &_value
		      );
    }
    else{
      std::cerr << "Undefined or invalid base. Specify hex or dec. Exiting..." << std::endl;
      exit(-1);
    }
    if ( _read == 5 ){
      _lines++;
      
      int _geomId = hAss.getGeomId(hAss.getSubdetector(std::string(_det))
			      , _eta, _phi, _dep);
      _cq.status = _value;
      _cq.onoff  = (_value & 65536)>>15;
      geomid_cq.insert(std::pair<int, HcalChannelQualityXml::ChannelQuality>(_geomId, _cq));      
      //std::cerr << "Line: " << _geomId << "   " << _cq.status << std::endl;
    } 
  }
  return _lines;
}


int HcalChannelQualityXml::writeStatusWordToStdout(std::string base){
  int _lines = 0;
  char _buf[128];
  int _detId = 0; // dummy as it is optional in the ASCII file
  sprintf(_buf, "#             eta             phi             dep             det    value      DetId");
  std::cout << _buf << std::endl;
  for (std::map<int,ChannelQuality>::const_iterator _cq = geomid_cq.begin();
       _cq != geomid_cq.end();
       _cq++){
    _lines++;
    _detId = hAss.getRawIdFromCmssw(_cq->first);
    if ( base.find("hex")!=std::string::npos ){
      sprintf(_buf," %16d%16d%16d%16s%16.8X%11.8X",
	      hAss.getHcalIeta(_cq->first),
	      hAss.getHcalIphi(_cq->first),
	      hAss.getHcalDepth(_cq->first),
	      hAss.getSubdetectorString(hAss.getHcalSubdetector(_cq->first)).c_str(),
	      _cq->second.status,
	      _detId
	      );
    }
    else if ( base.find("dec")!=std::string::npos ){
      sprintf(_buf," %16d%16d%16d%16s%16d%11.8d",
	      hAss.getHcalIeta(_cq->first),
	      hAss.getHcalIphi(_cq->first),
	      hAss.getHcalDepth(_cq->first),
	      hAss.getSubdetectorString(hAss.getHcalSubdetector(_cq->first)).c_str(),
	      _cq->second.status,
	      _detId
	      );
    }
    else{
      std::cerr << "Undefined or invalid base. Specify hex or dec. Exiting..." << std::endl;
      exit(-1);
    }
    std::cout << _buf << std::endl;
  }
  return _lines;
}


int HcalChannelQualityXml::makeXmlFromAsciiStream(int _runnumber,
						  int _iov_begin,
						  int _iov_end,
						  std::string _tag,
						  std::string _elements_comment,
						  std::string _base
						  )
{
  readStatusWordFromStdin(_base);
  set_header_run_number(_runnumber);
  set_elements_iov_begin(_iov_begin);
  set_elements_iov_end(_iov_end);
  set_elements_tag_name(_tag);
  set_elements_comment(_elements_comment);
  addChannelQualityGeom(geomid_cq);
  write();
  return 0;
}


int HcalChannelQualityXml::writeBaseLineFromOmdsToStdout(std::string _tag, int _iov_begin, std::string base){
  getBaseLineFromOmds(_tag, _iov_begin);
  writeStatusWordToStdout(base);
  return 0;
}



//
//_____ dumps a sorted list of tags to stdout, newest first
//
int HcalChannelQualityXml::dumpTagsFromOmdsToStdout(){
  std::vector<std::string> _tags = getTagsFromOmds();
  for (std::vector<std::string>::const_iterator tag=_tags.begin(); tag!=_tags.end(); tag++){
    std::cout << *tag << std::endl;
  }
  return _tags.size();
}

//
//_____ reads a sorted list of tags, newest first
//
std::vector<std::string> HcalChannelQualityXml::getTagsFromOmds(){
  std::vector<std::string> _tags;
  static ConnectionManager conn;
  conn.connect();
  std::string query = "select distinct tag_name, min(record_id) as mrid ";
  query            += "from ";
  query            += "cms_hcl_hcal_cond.v_hcal_channel_quality cq ";
  query            += "group by tag_name ";
  query            += "order by mrid desc ";
  int _n_tags = 0;
  try {
    oracle::occi::Statement* stmt = conn.getStatement(query);
    oracle::occi::ResultSet *rs = stmt->executeQuery();
    while (rs->next()) {
      _n_tags++;
      _tags.push_back( rs->getString(1) );
    }
  }
  catch (SQLException& e) {
    std::cerr << ::toolbox::toString("Oracle  exception : %s",e.getMessage().c_str()) << std::endl;
    XCEPT_RAISE(hcal::exception::ConfigurationDatabaseException,::toolbox::toString("Oracle  exception : %s",e.getMessage().c_str()));
  }
  conn.disconnect();
  return _tags;
}



//
//_____ reads a sorted list of Iovs for a given tag, newest on top
//
int HcalChannelQualityXml::dumpIovsFromOmdsToStdout(std::string tag){
  std::vector<int> _iovs = getIovsFromOmds(tag);
  for (std::vector<int>::const_iterator tag=_iovs.begin(); tag!=_iovs.end(); tag++){
    std::cout << *tag << std::endl;
  }
  return _iovs.size();
}

//
//_____ reads a sorted list of iovs, newest first
//
std::vector<int> HcalChannelQualityXml::getIovsFromOmds(std::string tag){
  std::vector<int> _iovs;
  static ConnectionManager conn;
  conn.connect();
  std::string query = "select distinct cq.interval_of_validity_begin, min(record_id) as mrid ";
  query            += "from ";
  query            += "cms_hcl_hcal_cond.v_hcal_channel_quality cq ";
  query            += "where ";
  query            += "tag_name=:1 ";
  query            += "group by cq.interval_of_validity_begin ";
  query            += "order by mrid desc ";
  int _n_iovs = 0;
  try {
    oracle::occi::Statement* stmt = conn.getStatement(query);
    stmt->setString(1,tag);
    oracle::occi::ResultSet *rs = stmt->executeQuery();
    while (rs->next()) {
      _n_iovs++;
      _iovs.push_back( rs->getInt(1) );
    }
  }
  catch (SQLException& e) {
    std::cerr << ::toolbox::toString("Oracle  exception : %s",e.getMessage().c_str()) << std::endl;
    XCEPT_RAISE(hcal::exception::ConfigurationDatabaseException,::toolbox::toString("Oracle  exception : %s",e.getMessage().c_str()));
  }
  conn.disconnect();
  return _iovs;
}

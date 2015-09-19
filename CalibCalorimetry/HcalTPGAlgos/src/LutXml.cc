// -*- C++ -*-
//
// Package:     CaloOnlineTools/HcalOnlineDb
// Class  :     LutXml
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Gena Kukartsev, kukarzev@fnal.gov
//         Created:  Tue Mar 18 14:30:20 CDT 2008
//

#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <iconv.h>
#include <sys/time.h>

#include "CalibCalorimetry/HcalTPGAlgos/interface/LutXml.h"
#include "CalibCalorimetry/HcalTPGAlgos/interface/XMLProcessor.h"
#include "FWCore/Utilities/interface/md5.h"
#include "CalibCalorimetry/HcalTPGAlgos/interface/HcalEmap.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"

using namespace std;
XERCES_CPP_NAMESPACE_USE
//
//_____ following removed as a xalan-c component_____________________
//
// xalan-c init
//#include <xalanc/Include/PlatformDefinitions.hpp>
//#include <xalanc/XPath/XPathEvaluator.hpp>
//#include <xercesc/framework/LocalFileInputSource.hpp>
//XALAN_USING_XERCES(XMLPlatformUtils)
//#include <xalanc/XSLT/XSLTInputSource.hpp>
//#include <xalanc/PlatformSupport/XSLException.hpp>
//#include <xalanc/DOMSupport/XalanDocumentPrefixResolver.hpp>
//#include <xalanc/XPath/XObject.hpp>
//#include <xalanc/XalanSourceTree/XalanSourceTreeDOMSupport.hpp>
//#include <xalanc/XalanSourceTree/XalanSourceTreeInit.hpp>
//#include <xalanc/XalanSourceTree/XalanSourceTreeParserLiaison.hpp>  
//using namespace xalanc;



/*
LutXml & LutXml::operator+=( const LutXml & other)
{
  DOMNodeList * _children = other.getDocumentConst()->getChildNodes();
  int _length = _children->getLength();
  std::cout << "Nodes added:" << _length << std::endl;
  DOMNode * _node;
  for(int i=0;i!=_length;i++){
    _node = _children->item(i)->cloneNode(true);
    this->getDocument()->getDocumentElement()->appendChild(_node);
  }
  return *this;
}
*/



LutXml::Config::_Config()
{
  ieta = -1000;
  iphi = -1000;
  depth = -1;
  crate = -1;
  slot = -1;
  topbottom = -1;
  fiber = -1;
  fiberchan = -1;
  lut_type = -1;
  creationtag = "default_tag";

  char timebuf[50];
  time_t _time = time( NULL );
  //time_t _time = 1193697120;
  //strftime( timebuf, 50, "%c", gmtime( &_time ) );
  strftime( timebuf, 50, "%Y-%m-%d %H:%M:%S", gmtime( &_time ) );
  creationstamp = timebuf;

  formatrevision = "default_revision";
  targetfirmware = "default_revision";
  generalizedindex = -1;
}

LutXml::LutXml() : XMLDOMBlock( "CFGBrickSet", 1 )
{
  init();
}


LutXml::LutXml(InputSource & _source ) : XMLDOMBlock( _source )
{
  init();
}


LutXml::LutXml( std::string filename ) : XMLDOMBlock( filename )
{
  init();
}


LutXml::~LutXml()
{
  //delete brickElem; // belongs to document that belongs to parser???
  XMLString::release(&root);
  XMLString::release(&brick);
  //delete lut_map;
}


void LutXml::init( void )
{
  root  = XMLString::transcode("CFGBrickSet");
  brick = XMLString::transcode("CFGBrick");
  brickElem = 0;
  //lut_map = 0;
}


std::vector<unsigned int> * LutXml::getLutFast( uint32_t det_id ){
   /*
  if (lut_map){
    return &(*lut_map)[det_id];
  }
  else{
    std::cerr << "LUT not found, null pointer is returned" << std::endl;
    return 0;
  }
  */
   if (lut_map.find(det_id) != lut_map.end()) return &(lut_map)[det_id];
   std::cerr << "LUT not found, null pointer is returned" << std::endl;
   return 0;
}

//
//_____ following removed as a xalan-c component_____________________
//
/*
std::vector<unsigned int> LutXml::getLut( int lut_type, int crate, int slot, int topbottom, int fiber, int fiber_channel ){
  std::vector<unsigned int> _lut;
  //write();

  std::string _context = "/CFGBrickSet";
  char buf[1024];
  sprintf(buf,
	  "CFGBrick[Parameter[@name='LUT_TYPE']=%d and Parameter[@name='CRATE']=%d and Parameter[@name='SLOT']=%d and Parameter[@name='TOPBOTTOM']=%d and Parameter[@name='FIBER']=%d and Parameter[@name='FIBERCHAN']=%d]/Data",
	  lut_type, crate, slot, topbottom, fiber, fiber_channel);
  std::string _expression(buf);

  std::cout << _expression << std::endl;

  const XObjectPtr theResult = eval_xpath(_context,_expression);

  if(theResult.null() == false){
    std::cout << std::endl << theResult->str() << std::endl;
  }

  const XalanDOMString & _string = theResult->str();
  int _string_length = _string.length();
  XalanVector<char> _str;
  _string.transcode(_str);
  unsigned int _base = 16;
  unsigned int _item=0;
  for (int i=0; i!=_string_length; i++){
    bool _range;
    char ch_cur = _str[i];
    if (_base==16) _range = (ch_cur>='0' and ch_cur<='9') || (ch_cur>='a' and ch_cur<='f') || (ch_cur>='A' and ch_cur<='F');
    else if (_base==10) _range = (ch_cur>='0' and ch_cur<='9');
    if ( _range ){
      if ( ch_cur>='a' and ch_cur<='f' ) ch_cur += 10-'a';
      else if ( ch_cur>='A' and ch_cur<='F' ) ch_cur += 10-'A';
      else if ( ch_cur>='0' and ch_cur<='9' ) ch_cur += -'0';
      _item = _item*_base;
      _item += ch_cur;
      bool last_digit = false;
      if ( (i+1)==_string_length ) last_digit=true;
      else{
	char ch_next = _str[i+1];
	bool _range_next;
	if (_base==16) _range_next = (ch_next>='0' and ch_next<='9') || (ch_next>='a' and ch_next<='f') || (ch_next>='A' and ch_next<='F');
	else if (_base==10) _range_next = (ch_next>='0' and ch_next<='9');
	if ( !_range_next ) last_digit=true;
      }
      if (last_digit){
	_lut.push_back(_item);
	_item=0;
      }
    }
  }

  std::cout << "### ";
  for (std::vector<unsigned int>::const_iterator l=_lut.begin(); l!=_lut.end(); l++){
    std::cout << *l << " ";
  }
  std::cout << std::endl << std::endl;

  return _lut;
}
*/


// checksums_xml is 0 by default
void LutXml::addLut( LutXml::Config & _config, XMLDOMBlock * checksums_xml )
{
  DOMElement * rootElem = document -> getDocumentElement();

  brickElem = document->createElement( XMLProcessor::_toXMLCh("CFGBrick") );
  rootElem->appendChild(brickElem);

  addParameter( "IETA", "int", _config.ieta );
  addParameter( "IPHI", "int", _config.iphi );
  addParameter( "CRATE", "int", _config.crate );
  addParameter( "SLOT", "int", _config.slot );
  addParameter( "TOPBOTTOM", "int", _config.topbottom );
  addParameter( "LUT_TYPE", "int", _config.lut_type );
  addParameter( "CREATIONTAG", "string", _config.creationtag );
  addParameter( "CREATIONSTAMP", "string", _config.creationstamp );
  addParameter( "FORMATREVISION", "string", _config.formatrevision );
  addParameter( "TARGETFIRMWARE", "string", _config.targetfirmware );
  addParameter( "GENERALIZEDINDEX", "int", _config.generalizedindex );
  addParameter( "CHECKSUM", "string", get_checksum( _config.lut ) );

  if(_config.lut_type==1){ // linearizer LUT
    addParameter( "FIBER", "int", _config.fiber );
    addParameter( "FIBERCHAN", "int", _config.fiberchan );
    addParameter( "DEPTH", "int", _config.depth );
    addData( "128", "hex", _config.lut );
  }
  else if(_config.lut_type==2){ // compression LUT
  addParameter( "SLB", "int", _config.fiber );
  addParameter( "SLBCHAN", "int", _config.fiberchan );
    addData( "1024", "hex", _config.lut );
  }
  else{
    std::cout << "Unknown LUT type...produced XML will be incorrect" << std::endl;
  }

  // if the pointer to the checksums XML was given,
  // add the checksum to it
  // checksums_xml is 0 unless explicitely given
  if ( checksums_xml ){
    add_checksum( checksums_xml->getDocument(), _config );
  }
}

DOMElement * LutXml::addData( std::string _elements, std::string _encoding, const std::vector<unsigned int>& _lut )
{
  DOMElement * child    = document -> createElement( XMLProcessor::_toXMLCh( "Data" ) );
  child -> setAttribute( XMLProcessor::_toXMLCh("elements"), XMLProcessor::_toXMLCh( _elements ) );
  child -> setAttribute( XMLProcessor::_toXMLCh("encoding"), XMLProcessor::_toXMLCh( _encoding ) );

  std::stringstream buf;

  for (std::vector<unsigned int>::const_iterator iter = _lut.begin();iter!=_lut.end();iter++){
    char buf2[8];
    sprintf(buf2,"%x",(*iter));
    buf << buf2 << " ";
    //buf << (*iter) << " ";
  }

  std::string _value = buf . str();

  DOMText * data_value = document -> createTextNode( XMLProcessor::_toXMLCh(_value));
  child -> appendChild( data_value );

  brickElem -> appendChild( child );

  return child;
}



DOMElement * LutXml::add_checksum( DOMDocument * parent, Config & config )
{
  //template
  //   <Data crate='0' slot='2' fpga='1' fiber='1' fiberchan='0' luttype='1' elements='1' encoding='hex'>c6cf91b39e3bff21623fb7366efda1fd</Data>
  //
  DOMElement * child    = parent -> createElement( XMLProcessor::_toXMLCh( "Data" ) );
  child -> setAttribute( XMLProcessor::_toXMLCh("crate"),     XMLProcessor::_toXMLCh( config.crate ) );
  child -> setAttribute( XMLProcessor::_toXMLCh("slot"),      XMLProcessor::_toXMLCh( config.slot ) );
  child -> setAttribute( XMLProcessor::_toXMLCh("fpga"),      XMLProcessor::_toXMLCh( config.topbottom ) );
  child -> setAttribute( XMLProcessor::_toXMLCh("fiber"),     XMLProcessor::_toXMLCh( config.fiber ) );
  child -> setAttribute( XMLProcessor::_toXMLCh("fiberchan"), XMLProcessor::_toXMLCh( config.fiberchan ) );
  child -> setAttribute( XMLProcessor::_toXMLCh("luttype"),   XMLProcessor::_toXMLCh( config.lut_type ) );
  child -> setAttribute( XMLProcessor::_toXMLCh("elements"),  XMLProcessor::_toXMLCh( "1" ) );
  child -> setAttribute( XMLProcessor::_toXMLCh("encoding"),  XMLProcessor::_toXMLCh( "hex" ) );
  DOMText * checksum_value = parent -> createTextNode( XMLProcessor::_toXMLCh( get_checksum(config.lut) ));
  child -> appendChild( checksum_value );

  parent -> getDocumentElement() -> appendChild( child );

  return child;
}



DOMElement * LutXml::addParameter( std::string _name, std::string _type, std::string _value )
{
  DOMElement * child    = document -> createElement( XMLProcessor::_toXMLCh( "Parameter" ) );
  child -> setAttribute( XMLProcessor::_toXMLCh("name"), XMLProcessor::_toXMLCh( _name ) );
  child -> setAttribute( XMLProcessor::_toXMLCh("type"), XMLProcessor::_toXMLCh( _type ) );
  DOMText * parameter_value = document -> createTextNode( XMLProcessor::_toXMLCh(_value));
  child -> appendChild( parameter_value );

  brickElem -> appendChild( child );

  return child;
}



DOMElement * LutXml::addParameter( std::string _name, std::string _type, int _value )
{
  char buf[128];
  sprintf(buf, "%d", _value);
  std::string str_value = buf;
  return addParameter( _name, _type, str_value );
}




std::string & LutXml::getCurrentBrick( void )
{
  return getString( brickElem );
}



// do MD5 checksum
std::string LutXml::get_checksum( std::vector<unsigned int> & lut )
{
  std::stringstream result;
  md5_state_t md5er;
  md5_byte_t digest[16];
  md5_init(&md5er);
  // linearizer LUT:
  if ( lut . size() == 128 ){
    unsigned char tool[2];
    for (int i=0; i<128; i++) {
      tool[0]=lut[i]&0xFF;
      tool[1]=(lut[i]>>8)&0xFF;
      md5_append(&md5er,tool,2);
    }
  }
  // compression LUT:
  else if ( lut . size() == 1024 ){
    unsigned char tool;
    for (int i=0; i<1024; i++) {
      tool=lut[i]&0xFF;
      md5_append(&md5er,&tool,1);
    }
  }
  else{
    std::cout << "ERROR: irregular LUT size, do not know how to compute checksum, exiting..." << std::endl;
    exit(-1);
  }
  md5_finish(&md5er,digest);
  for (int i=0; i<16; i++) result << std::hex << (((int)(digest[i]))&0xFF);

  //std::cout << "CHECKSUM: ";
  //std::cout << result . str();
  //std::cout << std::endl;

  return result . str();
}


int LutXml::test_access( std::string filename ){
  //create_lut_map();
  //std::cout << "Created map size: " << lut_map->size() << std::endl;
  std::cout << "Created map size: " << lut_map.size() << std::endl;

  struct timeval _t;
  gettimeofday( &_t, NULL );
  double _time =(double)(_t . tv_sec) + (double)(_t . tv_usec)/1000000.0;

  HcalEmap _emap("./backup/official_emap_v6.04_080905.txt");
  std::vector<HcalEmap::HcalEmapRow> & _map = _emap.get_map();
  std::cout << "HcalEmap contains " << _map . size() << " entries" << std::endl;

  int _counter=0;
  for (std::vector<HcalEmap::HcalEmapRow>::const_iterator row=_map.begin(); row!=_map.end(); row++){
    if (row->subdet=="HB"){
      HcalDetId det_id(HcalBarrel,row->ieta,row->iphi,row->idepth);
      uint32_t raw_id = det_id.rawId();
      std::vector<unsigned int> * l = getLutFast(raw_id);
      if (l) _counter++;
    }
    if (row->subdet=="HE"){
      HcalDetId det_id(HcalEndcap,row->ieta,row->iphi,row->idepth);
      uint32_t raw_id = det_id.rawId();
      std::vector<unsigned int> * l = getLutFast(raw_id);
      if (l) _counter++;
    }
    if (row->subdet=="HF"){
      HcalDetId det_id(HcalForward,row->ieta,row->iphi,row->idepth);
      uint32_t raw_id = det_id.rawId();
      std::vector<unsigned int> * l = getLutFast(raw_id);
      if (l) _counter++;
    }
    if (row->subdet=="HO"){
      HcalDetId det_id(HcalOuter,row->ieta,row->iphi,row->idepth);
      uint32_t raw_id = det_id.rawId();
      std::vector<unsigned int> * l = getLutFast(raw_id);
      if (l) _counter++;
    }
  }
  gettimeofday( &_t, NULL );
  std::cout << "access to " << _counter << " HCAL channels took: " << (double)(_t . tv_sec) + (double)(_t . tv_usec)/1000000.0 - _time << "sec" << std::endl;

  //std::cout << std::endl;
  //for (std::vector<unsigned int>::const_iterator i=l->begin();i!=l->end();i++){
  //  std::cout << *i << " ";
  //}
  //std::cout << std::endl;

  return 0;
}

//
//_____ following removed as a xalan-c component_____________________
//
/*
int LutXml::test_xpath( std::string filename ){
  // http://svn.apache.org/repos/asf/xalan/c/tags/Xalan-C_1_10_0/samples/SimpleXPathAPI/SimpleXPathAPI.cpp
  
  XMLProcessor::getInstance();

  read_xml_file_xalan( filename );

  std::string _context = "/CFGBrickSet";
  //std::string _expression = "CFGBrick[Parameter[@name='IETA']=-1 and Parameter[@name='IPHI']=19]/Data";
  std::string _expression = "CFGBrick[Parameter[@name='IETA']=-1]/Data";
  std::cout << _expression << std::endl;

  const XObjectPtr theResult = eval_xpath(_context,_expression);

  if(theResult.null() == false){
    std::cout << "Number of nodes: " << theResult->nodeset().getLength() << std::endl;
    
    std::cout << "The std::string value of the result is:"
	 << std::endl
	 << theResult->str()
	 << std::endl
	 << std::endl;
  }
  return 0;
}
*/

HcalSubdetector LutXml::subdet_from_crate(int crate, int eta, int depth){
  HcalSubdetector result;
  // HBHE: 0,1,4,5,10,11,14,15,17
  // HF: 2,9,12
  // HO: 3,6,7,13

  if (crate==2 || crate==9 || crate==12) result=HcalForward;
  else if (crate==3 || crate==6 || crate==7 || crate==13) result=HcalOuter;
  else if (crate==0 || crate==1 || crate==4 || crate==5 || crate==10 || crate==11 || crate==14 || crate==15 || crate==17){
    if (eta<16) result=HcalBarrel;
    else if (eta>16) result=HcalEndcap;
    else if (eta==16 && depth!=3) result=HcalBarrel;
    else if (eta==16 && depth==3) result=HcalEndcap;
    else{
      std::cerr << "Impossible to determine HCAL subdetector!!!" << std::endl;
      exit(-1);
    }
  }
  else{
    std::cerr << "Impossible to determine HCAL subdetector!!!" << std::endl;
    exit(-1);
  }

  return result;
}


int LutXml::a_to_i(char * inbuf){
  int result;
  sscanf(inbuf,"%d",&result);
  return result;
}

// organize all LUTs in XML into a map for fast access
//
// FIXME: uses hardcoded CRATE-to-subdetector mapping
// FIXME: it would be better to use some official map
//
int LutXml::create_lut_map( void ){
  //delete lut_map;
  lut_map.clear();
  //lut_map = new std::map<uint32_t,std::vector<unsigned int> >();

  if (document){
    //DOMElement * rootElem = 
    DOMNodeList * brick_list = document->getDocumentElement()->getElementsByTagName(brick);
    int n_of_bricks = brick_list->getLength();
    for(int i=0; i!=n_of_bricks; i++){
      DOMElement * aBrick = (DOMElement *)(brick_list->item(i));
      DOMNodeList * par_list = aBrick->getElementsByTagName(XMLString::transcode("Parameter"));
      int n_of_par = par_list->getLength();
      int ieta=-99;
      int iphi=-99;
      int depth=-99;
      int crate=-99;
      int lut_type=-99;
      HcalSubdetector subdet;
      for(int j=0; j!=n_of_par; j++){
	//std::cout << "DEBUG: i,j: " << i << ", " << j << std::endl;
	DOMElement * aPar = (DOMElement *)(par_list->item(j));
	char * aName = XMLString::transcode( aPar->getAttribute(XMLProcessor::_toXMLCh("name")) );
	if ( strcmp(aName, "IETA")==0 ) ieta=a_to_i(XMLString::transcode(aPar->getFirstChild()->getNodeValue()));
	if ( strcmp(aName, "IPHI")==0 ) iphi=a_to_i(XMLString::transcode(aPar->getFirstChild()->getNodeValue()));
	if ( strcmp(aName, "DEPTH")==0 ) depth=a_to_i(XMLString::transcode(aPar->getFirstChild()->getNodeValue()));
	if ( strcmp(aName, "CRATE")==0 ) crate=a_to_i(XMLString::transcode(aPar->getFirstChild()->getNodeValue()));
	if ( strcmp(aName, "LUT_TYPE")==0 ) lut_type=a_to_i(XMLString::transcode(aPar->getFirstChild()->getNodeValue()));
      }
      subdet=subdet_from_crate(crate,abs(ieta),depth);
      //std::cerr << "DEBUG: eta,phi,depth,crate,type,subdet: " << ieta << ", " << iphi << ", " << depth << ", " << crate << ", " << lut_type << ", " << subdet << std::endl;
      DOMElement * _data = (DOMElement *)(aBrick->getElementsByTagName(XMLString::transcode("Data"))->item(0));
      char * _str = XMLString::transcode(_data->getFirstChild()->getNodeValue());
      //std::cout << _str << std::endl;
      //
      // get the LUT vector
      int _string_length = strlen(_str);
      std::vector<unsigned int> _lut;
      unsigned int _base = 16;
      unsigned int _item=0;
      for (int i=0; i!=_string_length; i++){
	bool _range = false;
	char ch_cur = _str[i];
	if (_base==16) _range = (ch_cur>='0' and ch_cur<='9') || (ch_cur>='a' and ch_cur<='f') || (ch_cur>='A' and ch_cur<='F');
	else if (_base==10) _range = (ch_cur>='0' and ch_cur<='9');
	if ( _range ){
	  if ( ch_cur>='a' and ch_cur<='f' ) ch_cur += 10-'a';
	  else if ( ch_cur>='A' and ch_cur<='F' ) ch_cur += 10-'A';
	  else if ( ch_cur>='0' and ch_cur<='9' ) ch_cur += -'0';
	  _item = _item*_base;
	  _item += ch_cur;
	  bool last_digit = false;
	  if ( (i+1)==_string_length ) last_digit=true;
	  else{
	    char ch_next = _str[i+1];
	    bool _range_next = false;
	    if (_base==16) _range_next = (ch_next>='0' and ch_next<='9') || (ch_next>='a' and ch_next<='f') || (ch_next>='A' and ch_next<='F');
	    else if (_base==10) _range_next = (ch_next>='0' and ch_next<='9');
	    if ( !_range_next ) last_digit=true;
	  }
	  if (last_digit){
	    _lut.push_back(_item);
	    _item=0;
	  }
	}
      }
      // filling the map
      uint32_t _key = 0;
      if (lut_type==1){
	HcalDetId _id(subdet,ieta,iphi,depth);
	_key = _id.rawId();
      }
      else if (lut_type==2){
	HcalTrigTowerDetId _id(ieta,iphi);
	_key = _id.rawId();
      }
      lut_map.insert(std::pair<uint32_t,std::vector<unsigned int> >(_key,_lut));
    }
  }
  else{
    std::cerr << "XML file with LUTs is not loaded, cannot create map!" << std::endl;
  }



  return 0;
}

LutXml::const_iterator LutXml::begin() const{
   return lut_map.begin();
}

LutXml::const_iterator LutXml::end() const{
   return lut_map.end();
}

LutXml::const_iterator LutXml::find(uint32_t id) const{
   return lut_map.find(id);
}

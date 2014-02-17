// -*- C++ -*-
//
// Package:     XMLTools
// Class  :     XMLRBXPedestalsLoader
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Gena Kukartsev, kukarzev@fnal.gov
//         Created:  Tue Oct 23 14:30:20 CDT 2007
// $Id: XMLRBXPedestalsLoader.cc,v 1.8 2010/08/06 20:24:13 wmtan Exp $
//

// system include files

// user include files
#include "CaloOnlineTools/HcalOnlineDb/interface/XMLRBXPedestalsLoader.h"
#include "CalibCalorimetry/HcalTPGAlgos/interface/XMLProcessor.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/LMap.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
XMLRBXPedestalsLoader::loaderBaseConfig::_loaderBaseConfig()
{
  extention_table_name = "HCAL_RBX_CONFIGURATION_TYPE01";
  name = "HCAL RBX configuration [PEDESTAL]";
  run_mode = "auto";
  data_set_id = "-1";
  iov_id = "1";
  iov_begin = 1;
  iov_end = -1;
  tag_id = "2";
  tag_mode = "auto";
  tag_name = "test_kukartsev_v1";
  detector_name = "HCAL";
  elements_comment_description = "RBX pedestals";
}

XMLRBXPedestalsLoader::datasetDBConfig::_datasetDBConfig() : XMLProcessor::DBConfig()
{
  comment_description = "RBX pedestals for an RBX slot";
  name_label = "HEM10";
  kind_of_part = "HCAL RBX Slot";
}

XMLRBXPedestalsLoader::XMLRBXPedestalsLoader( loaderBaseConfig * config, std::string templateBase ) : XMLDOMBlock( templateBase )
{

  init();

  setTagValue( "EXTENSION_TABLE_NAME", config -> extention_table_name );
  setTagValue( "NAME", config -> name );
  setTagAttribute( "RUN", "mode", config -> run_mode );
  setTagAttribute( "DATA_SET", "id", config -> data_set_id );
  setTagAttribute( "IOV", "id", config -> iov_id );
  setTagValue( "INTERVAL_OF_VALIDITY_BEGIN", config -> iov_begin );
  setTagValue( "INTERVAL_OF_VALIDITY_END", config -> iov_end );
  setTagAttribute( "TAG", "id", config -> tag_id );
  setTagAttribute( "TAG", "mode", config -> tag_mode );
  setTagValue( "TAG_NAME", config -> tag_name );
  setTagValue( "DETECTOR_NAME", config -> detector_name );
  setTagValue( "COMMENT_DESCRIPTION", config -> elements_comment_description );
  setTagAttribute( "TAG", "idref", config -> tag_id, 1 );
  setTagAttribute( "IOV", "idref", config -> iov_id, 1 );
  setTagAttribute( "DATA_SET", "idref", config -> data_set_id, 1 );
}


XMLRBXPedestalsLoader::~XMLRBXPedestalsLoader()
{
  if( _data_ped_delay ) delete _data_ped_delay;
  if( _data_gol ) delete _data_gol;
  if( _data_led ) delete _data_led;
}


int XMLRBXPedestalsLoader::addRBXSlot( datasetDBConfig * config, std::string brickFileName, std::string rbx_config_type, std::string templateFileName )
{
  DOMElement * root = document -> getDocumentElement();

  XMLDOMBlock dataSetDoc( templateFileName );
  DOMDocument * dataSet = dataSetDoc . getDocument();

  char timebuf[50];
  strftime( timebuf, 50, "%Y-%m-%d %H:%M:%S.0", gmtime( &(config -> create_timestamp) ) );
  setTagValue( "CREATE_TIMESTAMP", timebuf , 0, dataSet );
  setTagValue( "CREATED_BY_USER", config -> created_by_user, 0, dataSet );
  setTagValue( "VERSION", config -> version, 0, dataSet );
  setTagValue( "SUBVERSION", config -> subversion, 0, dataSet );
  setTagValue( "COMMENT_DESCRIPTION", config -> comment_description, 0, dataSet );
  setTagValue( "NAME_LABEL", config -> name_label, 0, dataSet );
  setTagValue( "KIND_OF_PART", config -> kind_of_part, 0, dataSet );

  // <DATA/>
  //XMLDOMBlock rbxBrickDoc( "rbx_HBM01_PEDESTAL.xml" );
  XMLDOMBlock rbxBrickDoc( brickFileName );
  DOMDocument * rbxBrick = rbxBrickDoc . getDocument();
  if ( rbx_config_type == "pedestals" || rbx_config_type == "delays" || rbx_config_type == "gols" ){
    for ( unsigned int _item = 0; _item < rbxBrick -> getElementsByTagName( XMLProcessor::_toXMLCh( "Data" ) ) -> getLength(); _item++ ){
      DOMElement * dataset_root = dataSet -> getDocumentElement();
      
      std::string _rm;
      std::string _qie;
      std::string _adc;
      //string _led_item;
      
      MemBufInputSource * _data; // a container for the XML template for a data block
      if ( rbx_config_type == "pedestals" || rbx_config_type == "delays" ){
	_rm = rbxBrickDoc . getTagAttribute( "Data", "rm", _item );
	_qie = rbxBrickDoc . getTagAttribute( "Data", "card", _item );
	_adc = rbxBrickDoc . getTagAttribute( "Data", "qie", _item );
	_data = _data_ped_delay;
      }
      else if ( rbx_config_type == "gols" ){
	_rm = rbxBrickDoc . getTagAttribute( "Data", "rm", _item );
	_qie = rbxBrickDoc . getTagAttribute( "Data", "card", _item );
	_adc = rbxBrickDoc . getTagAttribute( "Data", "gol", _item );
	_data = _data_gol;
      }
      //else if ( rbx_config_type == "leds" ){
      //_led_item = rbxBrickDoc . getTagAttribute( "Data", "item", _item );
      //_data = _data_led;
      //}
      else{
	std::cout << "XMLRBXPedestalsLoader::addRBXSlot(): Unknown config type... exiting" << std::endl;
	exit(1);
      }
      XMLDOMBlock dataDoc( *_data );
      DOMDocument * data = dataDoc . getDocument();
      std::string _value = rbxBrickDoc . getTagValue( "Data", _item );
      if ( rbx_config_type == "pedestals" || rbx_config_type == "delays" ){
	setTagValue( "MODULE_POSITION", _rm, 0, data );
	setTagValue( "QIE_CARD_POSITION", _qie, 0, data );
	setTagValue( "QIE_ADC_NUMBER", _adc, 0, data );
      }
      else if ( rbx_config_type == "gols" ){
	setTagValue( "MODULE_POSITION", _rm, 0, data );
	setTagValue( "QIE_CARD_POSITION", _qie, 0, data );
	setTagValue( "FIBER_NUMBER", _adc, 0, data );
      }
      else if ( rbx_config_type == "gols" ){
	setTagValue( "MODULE_POSITION", _rm, 0, data );
	setTagValue( "QIE_CARD_POSITION", _qie, 0, data );
	setTagValue( "FIBER_NUMBER", _adc, 0, data );
      }
      else if ( rbx_config_type == "leds" ){
	setTagValue( "MODULE_POSITION", _rm, 0, data );
	setTagValue( "QIE_CARD_POSITION", _qie, 0, data );
	setTagValue( "FIBER_NUMBER", _adc, 0, data );
      }
      else{
	std::cout << "XMLRBXPedestalsLoader::addRBXSlot(): Unknown config type... exiting" << std::endl;
	exit(1);
      }
      setTagValue( "INTEGER_VALUE", _value, 0, data );
      DOMNode * cloneData = dataSet -> importNode( data -> getDocumentElement(), true );
      dataset_root -> appendChild( cloneData );
    }
  }
  else if ( rbx_config_type == "leds" ){
    DOMElement * dataset_root = dataSet -> getDocumentElement();
    
    std::string _led_item;
    
    MemBufInputSource * _data; // a container for the XML template for a data block
    _data = _data_led;
    std::string _value;

    XMLDOMBlock dataDoc( *_data );
    DOMDocument * data = dataDoc . getDocument();

    int _item = 0;
    _led_item = rbxBrickDoc . getTagAttribute( "Data", "item", _item );
    // FIXME: need to check that the right data tag (_led_item) from the original brick is being processed
    _value = rbxBrickDoc . getTagValue( "Data", _item );
    setTagValue( "LED1_ON_IS_CHECKED", _value, 0, data );
    setTagValue( "SET_LEDS_IS_CHECKED", _value, 0, data );
    _item = 1;
    _led_item = rbxBrickDoc . getTagAttribute( "Data", "item", _item );
    _value = rbxBrickDoc . getTagValue( "Data", _item );
    setTagValue( "LED2_ON_IS_CHECKED", _value, 0, data );
    if (_value.find("0")==std::string::npos){
      setTagValue( "SET_LEDS_IS_CHECKED", _value, 0, data );
    }
    _item = 2;
    _led_item = rbxBrickDoc . getTagAttribute( "Data", "item", _item );
    _value = rbxBrickDoc . getTagValue( "Data", _item );
    setTagValue( "LED_AMPLITUDE", _value, 0, data );
    _item = 3;
    _led_item = rbxBrickDoc . getTagAttribute( "Data", "item", _item );
    _value = rbxBrickDoc . getTagValue( "Data", _item );
    setTagValue( "BUNCH_NUMBER", _value, 0, data );
    
    DOMNode * cloneData = dataSet -> importNode( data -> getDocumentElement(), true );
    dataset_root -> appendChild( cloneData );
  }

  // copy the <data_set> node into the final XML
  DOMNode * cloneDataSet = document -> importNode( dataSet -> getDocumentElement(), true );
  root -> appendChild( cloneDataSet );

  // update header from the brick
  std::string _name;
  int parameter_iter = -1;
  do
    {
      parameter_iter++;
      _name = rbxBrickDoc . getTagAttribute( "Parameter", "name", parameter_iter );
      //std::cout << _name << std::endl;
    } while( _name != "CREATIONTAG" ); 
  std::string _creationtag = rbxBrickDoc . getTagValue( "Parameter", parameter_iter );
  //std::cout << _creationtag << std::endl;
  //setTagValue( "TAG_NAME", _creationtag );    // uncomment if want to pick up tag name from the brick set

  parameter_iter = -1;
  do
    {
      parameter_iter++;
      _name = rbxBrickDoc . getTagAttribute( "Parameter", "name", parameter_iter );
      //std::cout << _name << std::endl;
    } while( _name != "RBX" ); 
  std::string _rbx = rbxBrickDoc . getTagValue( "Parameter", parameter_iter );

  // name_label fix - should be temporary
  //fixRbxName( _rbx );
  std::cout << _rbx << std::endl;
  setTagValue( "NAME_LABEL", _rbx );  

  // change kind of part name if this is ZDC (there is only one - ZDC01)
  if (_rbx.find("ZDC01")!=std::string::npos){
    setTagValue( "KIND_OF_PART", "HCAL ZDC RBX" );
    std::cout << " --> ZDC RBX!" << std::endl;
  }

  return 0;
}

int XMLRBXPedestalsLoader::fixRbxName( std::string & rbx )
{
  std::string _fourth = rbx . substr(0,4);
  if ( _fourth == "HBM0" || _fourth == "HBP0" || _fourth == "HEP0" || _fourth == "HFP0" || _fourth == "HOP0" || _fourth == "HOM0" ) rbx . erase( 3, 1 );

  if ( rbx == "HO0M02" ) rbx = "HO0M2";
  if ( rbx == "HO0M04" ) rbx = "HO0M4";
  if ( rbx == "HO0M06" ) rbx = "HO0M6";
  if ( rbx == "HO0M08" ) rbx = "HO0M8";
  if ( rbx == "HO0M10" ) rbx = "HO0M10";
  if ( rbx == "HO0M12" ) rbx = "HO0M12";
  if ( rbx == "HO0P02" ) rbx = "HO0P2";
  if ( rbx == "HO0P04" ) rbx = "HO0P4";
  if ( rbx == "HO0P06" ) rbx = "HO0P6";
  if ( rbx == "HO0P08" ) rbx = "HO0P8";
  if ( rbx == "HO0P10" ) rbx = "HO0P10";
  if ( rbx == "HO0P12" ) rbx = "HO0P12";

  return 0;
}

int XMLRBXPedestalsLoader::init( void )
{
  // define the <DATA/> template for pedestals and zero delays
  static const char * _str =  "\
  <DATA>\n\
   <MODULE_POSITION>2</MODULE_POSITION>\n\
   <QIE_CARD_POSITION>1</QIE_CARD_POSITION>\n\
   <QIE_ADC_NUMBER>0</QIE_ADC_NUMBER>\n\
   <INTEGER_VALUE>4</INTEGER_VALUE>\n\
  </DATA>\n\
  ";
  const XMLByte * _template = (const XMLByte *)_str;
  _data_ped_delay = new MemBufInputSource( _template, strlen( (const char *)_template ), "_data_ped_delay", false );

  // define the <DATA/> template for gol currents
  static const char * _str2 =  "\
  <DATA>\n\
   <MODULE_POSITION>2</MODULE_POSITION>\n\
   <QIE_CARD_POSITION>1</QIE_CARD_POSITION>\n\
   <FIBER_NUMBER>0</FIBER_NUMBER>\n\
   <INTEGER_VALUE>4</INTEGER_VALUE>\n\
  </DATA>\n\
  ";
  const XMLByte * _template2 = (const XMLByte *)_str2;
  _data_gol = new MemBufInputSource( _template2, strlen( (const char *)_template2 ), "_data_gol", false );

  // define the <DATA/> template for LED data
  static const char * _str3 =  "\
  <DATA>\n\
   <RM1_IS_CHECKED>1</RM1_IS_CHECKED>\n\
   <RM2_IS_CHECKED>1</RM2_IS_CHECKED>\n\
   <RM3_IS_CHECKED>1</RM3_IS_CHECKED>\n\
   <RM4_IS_CHECKED>1</RM4_IS_CHECKED>\n\
   <CALIB_IS_CHECKED>1</CALIB_IS_CHECKED>\n\
   <RESET1_IS_CHECKED>1</RESET1_IS_CHECKED>\n\
   <RESET1_VALUE>40</RESET1_VALUE>\n\
   <RESET1_WAIT_CYCLES>3</RESET1_WAIT_CYCLES>\n\
   <RESET2_IS_CHECKED>1</RESET2_IS_CHECKED>\n\
   <RESET2_VALUE>40</RESET2_VALUE>\n\
   <RESET2_WAIT_CYCLES>3</RESET2_WAIT_CYCLES>\n\
   <SET_LEDS_IS_CHECKED>0</SET_LEDS_IS_CHECKED>\n\
   <LED1_ON_IS_CHECKED>0</LED1_ON_IS_CHECKED>\n\
   <LED2_ON_IS_CHECKED>0</LED2_ON_IS_CHECKED>\n\
   <LED_AMPLITUDE>128</LED_AMPLITUDE>\n\
   <LED_DELAY>1000</LED_DELAY>\n\
   <TTCRX_PHASE>100</TTCRX_PHASE>\n\
   <BROADCAST_SETTINGS>1</BROADCAST_SETTINGS>\n\
   <QIE_RESET_DELAY>5</QIE_RESET_DELAY>\n\
   <BUNCH_NUMBER>2000</BUNCH_NUMBER>\n\
  </DATA>\n\
  ";
  const XMLByte * _template3 = (const XMLByte *)_str3;
  _data_led = new MemBufInputSource( _template3, strlen( (const char *)_template3 ), "_data_led", false );

  return 0;
}

//
// const member functions
//

//
// static member functions
//

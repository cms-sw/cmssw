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
// $Id: XMLRBXPedestalsLoader.cc,v 1.1 2007/12/06 02:26:38 kukartse Exp $
//

// system include files

// user include files
#include "CaloOnlineTools/HcalOnlineDb/interface/XMLRBXPedestalsLoader.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/XMLProcessor.h"
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

XMLRBXPedestalsLoader::XMLRBXPedestalsLoader( loaderBaseConfig * config, string templateBase ) : XMLDOMBlock( templateBase )
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
  if( _data ) delete _data;
}


int XMLRBXPedestalsLoader::addRBXSlot( datasetDBConfig * config, string brickFileName, string templateFileName )
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
  for ( int _item = 0; _item < rbxBrick -> getElementsByTagName( XMLProcessor::_toXMLCh( "Data" ) ) -> getLength(); _item++ )
    {
      DOMElement * dataset_root = dataSet -> getDocumentElement();
      XMLDOMBlock dataDoc( *_data );
      DOMDocument * data = dataDoc . getDocument();
      string _rm = rbxBrickDoc . getTagAttribute( "Data", "rm", _item );
      string _qie = rbxBrickDoc . getTagAttribute( "Data", "card", _item );
      string _adc= rbxBrickDoc . getTagAttribute( "Data", "qie", _item );
      string _value = rbxBrickDoc . getTagValue( "Data", _item );
      setTagValue( "MODULE_POSITION", _rm, 0, data );
      setTagValue( "QIE_CARD_POSITION", _qie, 0, data );
      setTagValue( "QIE_ADC_NUMBER", _adc, 0, data );
      setTagValue( "INTEGER_VALUE", _value, 0, data );
      DOMNode * cloneData = dataSet -> importNode( data -> getDocumentElement(), true );
      dataset_root -> appendChild( cloneData );
    }

  // copy the <data_set> node into the final XML
  DOMNode * cloneDataSet = document -> importNode( dataSet -> getDocumentElement(), true );
  root -> appendChild( cloneDataSet );

  // update header from the brick
  string _name;
  int parameter_iter = -1;
  do
    {
      parameter_iter++;
      _name = rbxBrickDoc . getTagAttribute( "Parameter", "name", parameter_iter );
      //cout << _name << endl;
    } while( _name != "CREATIONTAG" ); 
  string _creationtag = rbxBrickDoc . getTagValue( "Parameter", parameter_iter );
  //cout << _creationtag << endl;
  //setTagValue( "TAG_NAME", _creationtag );    // uncomment if want to pick up tag name from the brick set

  parameter_iter = -1;
  do
    {
      parameter_iter++;
      _name = rbxBrickDoc . getTagAttribute( "Parameter", "name", parameter_iter );
      //cout << _name << endl;
    } while( _name != "RBX" ); 
  string _rbx = rbxBrickDoc . getTagValue( "Parameter", parameter_iter );

  // name_label fix - should be temporary
  //fixRbxName( _rbx );
  cout << _rbx << endl;
  setTagValue( "NAME_LABEL", _rbx );  

  return 0;
}

int XMLRBXPedestalsLoader::fixRbxName( string & rbx )
{
  string _fourth = rbx . substr(0,4);
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
  // define the <DATA/> template
  static const char * _str =  "\\
  <DATA>\n\\
   <MODULE_POSITION>2</MODULE_POSITION>\n\\
   <QIE_CARD_POSITION>1</QIE_CARD_POSITION>\n\\
   <QIE_ADC_NUMBER>0</QIE_ADC_NUMBER>\n\\
   <INTEGER_VALUE>4</INTEGER_VALUE>\n\\
  </DATA>\n\\
  ";
  const XMLByte * _template = (const XMLByte *)_str;
  _data = new MemBufInputSource( _template, strlen( (const char *)_template ), "_data", false );

  return 0;
}

//
// const member functions
//

//
// static member functions
//

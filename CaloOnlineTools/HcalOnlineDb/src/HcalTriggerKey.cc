// -*- C++ -*-
//
// Package:     XMLTools
// Class  :     HcalTriggerKey
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Gena Kukartsev, kukarzev@fnal.gov
//         Created:  August 27, 2008
//

#include "CaloOnlineTools/HcalOnlineDb/interface/HcalTriggerKey.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/XMLProcessor.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/LMap.h"

HcalTriggerKey::HcalTriggerKey( void )
{
  _root = 0;
  _data = 0;
  init();
  parse(*_root);
}


HcalTriggerKey::~HcalTriggerKey()
{
  if( _root ) delete _root;
  if( _data ) delete _data;
}


int HcalTriggerKey::init( void )
{
  static const char * _str =  "\
  <ROOT>\n\
    <HEADER>\n\
      <TYPE>\n\
        <EXTENSION_TABLE_NAME>HCAL_TRIGGER_KEY</EXTENSION_TABLE_NAME>\n\
        <NAME>HCAL trigger key</NAME>\n\
      </TYPE>\n\
       <RUN mode='no-run' />\n\
    </HEADER>\n\
   <DATA_SET>\n\
      <VERSION>test_version</VERSION>\n\
      <SUBVERSION>1</SUBVERSION>\n\
     <CREATE_TIMESTAMP>2000-01-01 00:00:00.0</CREATE_TIMESTAMP>\n\
     <CREATED_BY_USER>kukarzev</CREATED_BY_USER>\n\
     <COMMENT_DESCRIPTION>test trigger key</COMMENT_DESCRIPTION>\n\
      <PART>\n\
         <NAME_LABEL>CMS-HCAL-ROOT</NAME_LABEL>\n\
      </PART>\n\
   </DATA_SET>\n\
  <!-- Tags secton -->\n\
      <ELEMENTS>\n\
          <DATA_SET id='-1'/>\n\
          <IOV id='1'>\n\
              <INTERVAL_OF_VALIDITY_BEGIN>0</INTERVAL_OF_VALIDITY_BEGIN>\n\
              <INTERVAL_OF_VALIDITY_END>1</INTERVAL_OF_VALIDITY_END>\n\
          </IOV>\n\
          <TAG id='2' mode='auto'>\n\
              <TAG_NAME>trigger_key_tag</TAG_NAME>\n\
              <DETECTOR_NAME>HCAL</DETECTOR_NAME>\n\
              <COMMENT_DESCRIPTION>trigger_key_tag</COMMENT_DESCRIPTION>\n\
          </TAG>\n\
      </ELEMENTS>\n\
      <MAPS>\n\
          <TAG idref='2'>\n\
              <IOV idref='1'>\n\
                  <DATA_SET idref='-1' />\n\
              </IOV>\n\
          </TAG>\n\
      </MAPS>\n\
  </ROOT>\n\
  ";
  const XMLByte * _template = (const XMLByte *)_str;
  _root = new MemBufInputSource( _template, strlen( (const char *)_template ), "_root", false );

  static const char * _str2 =  "\
      <DATA>\n\
         <TRIGGER_KEY_ID>TEST_trigger_key</TRIGGER_KEY_ID>\n\
         <TRIGGER_KEY_CONFIG_TYPE>LUT_tag</TRIGGER_KEY_CONFIG_TYPE>\n\
         <TRIGGER_KEY_CONFIG_VALUE>CRUZET_part4_physics_v3</TRIGGER_KEY_CONFIG_VALUE>\n\
      </DATA>\n\
  ";
  const XMLByte * _template2 = (const XMLByte *)_str2;
  _data = new MemBufInputSource( _template2, strlen( (const char *)_template2 ), "_data", false );

  return 0;
}


int HcalTriggerKey::add_data( string id, string type, string value){
  XMLDOMBlock data_block( *_data );
  DOMDocument * data_doc = data_block . getDocument();

  DOMElement * data_set_elem = (DOMElement *)(document -> getElementsByTagName( XMLProcessor::_toXMLCh( "DATA_SET" ) ) -> item(0));

  DOMNode * clone_data = document -> importNode( data_doc -> getDocumentElement(), true );
  data_set_elem -> appendChild( clone_data );

  write("stdout");

  return 0;
}

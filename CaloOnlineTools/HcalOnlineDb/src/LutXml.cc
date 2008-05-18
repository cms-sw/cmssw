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
// $Id: LutXml.cc,v 1.2 2008/04/16 13:31:25 kukartse Exp $
//

// system include files
#include "CaloOnlineTools/HcalOnlineDb/interface/LutXml.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/XMLProcessor.h"

#include <iostream>
#include <string>
#include <vector>
#include <sstream>

using namespace std;

/*
LutXml & LutXml::operator+=( const LutXml & other)
{
  DOMNodeList * _children = other.getDocumentConst()->getChildNodes();
  int _length = _children->getLength();
  cout << "Children nodes:" << _length << endl;
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


LutXml::~LutXml()
{

}


void LutXml::init( void )
{
  brickElem = NULL;
}


void LutXml::addLut( LutXml::Config & _config )
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
    cout << "Unknown LUT type...produced XML will be incorrect" << endl;
  }
}

DOMElement * LutXml::addData( string _elements, string _encoding, std::vector<unsigned int> _lut )
{
  DOMElement * child    = document -> createElement( XMLProcessor::_toXMLCh( "Data" ) );
  child -> setAttribute( XMLProcessor::_toXMLCh("elements"), XMLProcessor::_toXMLCh( _elements ) );
  child -> setAttribute( XMLProcessor::_toXMLCh("encoding"), XMLProcessor::_toXMLCh( _encoding ) );

  stringstream buf;

  for (std::vector<unsigned int>::const_iterator iter = _lut.begin();iter!=_lut.end();iter++){
    char buf2[8];
    sprintf(buf2,"%x",(*iter));
    buf << buf2 << " ";
    //buf << (*iter) << " ";
  }

  string _value = buf . str();

  DOMText * data_value = document -> createTextNode( XMLProcessor::_toXMLCh(_value));
  child -> appendChild( data_value );

  brickElem -> appendChild( child );

  return child;
}



DOMElement * LutXml::addParameter( string _name, string _type, string _value )
{
  DOMElement * child    = document -> createElement( XMLProcessor::_toXMLCh( "Parameter" ) );
  child -> setAttribute( XMLProcessor::_toXMLCh("name"), XMLProcessor::_toXMLCh( _name ) );
  child -> setAttribute( XMLProcessor::_toXMLCh("type"), XMLProcessor::_toXMLCh( _type ) );
  DOMText * parameter_value = document -> createTextNode( XMLProcessor::_toXMLCh(_value));
  child -> appendChild( parameter_value );

  brickElem -> appendChild( child );

  return child;
}



DOMElement * LutXml::addParameter( string _name, string _type, int _value )
{
  char buf[128];
  sprintf(buf, "%d", _value);
  string str_value = buf;
  return addParameter( _name, _type, str_value );
}




std::string & LutXml::getCurrentBrick( void )
{
  return getString( brickElem );
}

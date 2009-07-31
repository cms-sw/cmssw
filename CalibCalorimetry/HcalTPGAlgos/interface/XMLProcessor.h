#ifndef HCALConfigDBTools_XMLTools_XMLProcessor_h
#define HCALConfigDBTools_XMLTools_XMLProcessor_h
// -*- C++ -*-
//
// Package:     XMLTools
// Class  :     XMLProcessor
// 
/**\class XMLProcessor XMLProcessor.h CalibCalorimetry/HcalTPGAlgos/interface/XMLProcessor.h

 Description: Testing Xerces library for processing HCAL DB XML wrappers

 Usage:
    <usage>

*/
//
// Original Author:  Gena Kukartsev
//         Created:  Sun Sep 23 16:57:06 CEST 2007
// $Id: XMLProcessor.h,v 1.2 2009/05/08 23:26:51 elmer Exp $
//

// system include files
#include <vector>
#include <string>
#include <time.h>
#include <xercesc/dom/DOM.hpp>
#include <xercesc/util/XMLString.hpp>
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/dom/DOMNode.hpp>
#include <xercesc/framework/MemBufInputSource.hpp>
#include <cstdio>

#if defined(XERCES_NEW_IOSTREAMS)
#include <iostream>
#else
#include <iostream.h>
#endif

#include "CalibCalorimetry/HcalTPGAlgos/interface/XMLDOMBlock.h"


XERCES_CPP_NAMESPACE_USE 
using namespace std;

class XMLProcessor
{
  
 public:
  
  typedef struct _loaderBaseConfig
  {
    _loaderBaseConfig();
    string extention_table_name;
    string name;
    string run_mode;
    string data_set_id;
    string iov_id;
    string iov_begin;
    string iov_end;
    string tag_id;
    string tag_mode;
    string tag_name;
    string detector_name;
    string comment_description;
  } loaderBaseConfig;

  typedef struct _LMapRowHBEF
  {
    int    side;
    int    eta;
    int    phi;
    int    dphi;
    int    depth;
    string det;
    string rbx;
    int    wedge;
    int    rm;
    int    pixel;
    int    qie;
    int    adc;
    int    rm_fi;
    int    fi_ch;
    int    crate;
    int    htr;
    string fpga;
    int    htr_fi;
    int    dcc_sl;
    int    spigo;
    int    dcc;
    int    slb;
    string slbin;
    string slbin2;
    string slnam;
    int    rctcra;
    int    rctcar;
    int    rctcon;
    string rctnam;
    int    fedid;
  } LMapRowHBEF;

  typedef struct _LMapRowHO
  {
    int    sideO;
    int    etaO;
    int    phiO;
    int    dphiO;
    int    depthO;
    string detO;
    string rbxO;
    int    sectorO;
    int    rmO;
    int    pixelO;
    int    qieO;
    int    adcO;
    int    rm_fiO;
    int    fi_chO;
    string let_codeO;
    int    crateO;
    int    htrO;
    string fpgaO;
    int    htr_fiO;
    int    dcc_slO;
    int    spigoO;
    int    dccO;
    int    fedidO;
  } LMapRowHO;

  typedef struct _DBConfig
  {
    _DBConfig();
    string    version;
    string    subversion;
    time_t create_timestamp;
    string created_by_user;
  } DBConfig;

  typedef struct _lutDBConfig : public _DBConfig
  {
    int    crateNumber;
  } lutDBConfig;
  
  typedef struct _checksumsDBConfig : public _DBConfig
  {
    string comment;
  } checksumsDBConfig;

  // this class is a singleton
  static XMLProcessor * getInstance()
    {
      if (!instance) instance = new XMLProcessor();
      return instance;
    }
  
  // returns XML string if target == "string" otherwise NULL
  XMLCh * serializeDOM( DOMNode* node, string target = "stdout" );

  inline static XMLCh * _toXMLCh( std::string temp );
  inline static XMLCh * _toXMLCh( int temp );
  inline static XMLCh * _toXMLCh( time_t temp );
  virtual ~XMLProcessor();
  
  int test( void );
  int init( void );
  int terminate( void );
  
  XMLDOMBlock * createLMapHBEFXMLBase( string templateFileName );
  XMLDOMBlock * createLMapHOXMLBase( string templateFileName );

  int addLMapHBEFDataset( XMLDOMBlock * doc, LMapRowHBEF * row, string templateFileName );
  int addLMapHODataset( XMLDOMBlock * doc, LMapRowHO * row, string templateFileName );
 
  int write( XMLDOMBlock * doc, string target = "stdout" );
  
 private:
  XMLProcessor();

  XMLProcessor(const XMLProcessor&); // stop default
  
  //const XMLProcessor& operator=(const XMLProcessor&); // stop default
  
  // ---------- member data --------------------------------
  static XMLProcessor * instance;
};

inline XMLCh* XMLProcessor::_toXMLCh( std::string temp )
{
  XMLCh* buff = XMLString::transcode(temp.c_str());    
  return  buff;
}

inline XMLCh* XMLProcessor::_toXMLCh( int temp )
{
  char buf[100];
  int status = snprintf( buf, 100, "%d", temp );
  if ( status >= 100 )
    {
      cout << "XMLProcessor::_toXMLCh(int temp): buffer overflow, the string will be truncated!" << endl;
    }
  else if ( status <0 )
    {
      cout << "XMLProcessor::_toXMLCh(int temp): output error" << endl;
    }
  XMLCh* buff = XMLString::transcode( buf );    
  return  buff;
}

inline XMLCh* XMLProcessor::_toXMLCh( double temp )
{
  char buf[100];
  int status = snprintf( buf, 100, "%e", temp );
  if ( status >= 100 )
    {
      cout << "XMLProcessor::_toXMLCh(int temp): buffer overflow, the string will be truncated!" << endl;
    }
  else if ( status <0 )
    {
      cout << "XMLProcessor::_toXMLCh(int temp): output error" << endl;
    }
  XMLCh* buff = XMLString::transcode( buf );    
  return  buff;
}

inline XMLCh* XMLProcessor::_toXMLCh( time_t temp )
{
  char buf[100];
  int status = strftime( buf, 50, "%c", gmtime( &temp ) );
  if ( status == 0 )
    {
      cout << "XML  Processor::_toXMLCh(int temp): buffer overflow, the string is indeterminate!" << endl;
    }
  XMLCh* buff = XMLString::transcode( buf );    
  return  buff;
}


#endif


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
// $Id: XMLProcessor.h,v 1.6 2010/08/06 20:24:02 wmtan Exp $
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

class XMLProcessor
{
  
 public:
  
  typedef struct _loaderBaseConfig
  {
    _loaderBaseConfig();
    std::string extention_table_name;
    std::string name;
    std::string run_mode;
    std::string data_set_id;
    std::string iov_id;
    std::string iov_begin;
    std::string iov_end;
    std::string tag_id;
    std::string tag_mode;
    std::string tag_name;
    std::string detector_name;
    std::string comment_description;
  } loaderBaseConfig;

  typedef struct _LMapRowHBEF
  {
    int    side;
    int    eta;
    int    phi;
    int    dphi;
    int    depth;
    std::string det;
    std::string rbx;
    int    wedge;
    int    rm;
    int    pixel;
    int    qie;
    int    adc;
    int    rm_fi;
    int    fi_ch;
    int    crate;
    int    htr;
    std::string fpga;
    int    htr_fi;
    int    dcc_sl;
    int    spigo;
    int    dcc;
    int    slb;
    std::string slbin;
    std::string slbin2;
    std::string slnam;
    int    rctcra;
    int    rctcar;
    int    rctcon;
    std::string rctnam;
    int    fedid;
  } LMapRowHBEF;

  typedef struct _LMapRowHO
  {
    int    sideO;
    int    etaO;
    int    phiO;
    int    dphiO;
    int    depthO;
    std::string detO;
    std::string rbxO;
    int    sectorO;
    int    rmO;
    int    pixelO;
    int    qieO;
    int    adcO;
    int    rm_fiO;
    int    fi_chO;
    std::string let_codeO;
    int    crateO;
    int    htrO;
    std::string fpgaO;
    int    htr_fiO;
    int    dcc_slO;
    int    spigoO;
    int    dccO;
    int    fedidO;
  } LMapRowHO;

  typedef struct _DBConfig
  {
    _DBConfig();
    std::string    version;
    std::string    subversion;
    time_t create_timestamp;
    std::string created_by_user;
  } DBConfig;

  typedef struct _lutDBConfig : public _DBConfig
  {
    int    crateNumber;
  } lutDBConfig;
  
  typedef struct _checksumsDBConfig : public _DBConfig
  {
    std::string comment;
  } checksumsDBConfig;

  // this class is a singleton
  static XMLProcessor * getInstance()
    {
      if (!instance) instance = new XMLProcessor();
      return instance;
    }
  
  // returns XML std::string if target == "string" otherwise NULL
  XMLCh * serializeDOM( DOMNode* node, std::string target = "stdout" );

  inline static XMLCh * _toXMLCh( std::string temp );
  inline static XMLCh * _toXMLCh( int temp );
  inline static XMLCh * _toXMLCh( double temp );
  inline static XMLCh * _toXMLCh( time_t temp );
  virtual ~XMLProcessor();
  
  int test( void );
  int init( void );
  int terminate( void );
  
  XMLDOMBlock * createLMapHBEFXMLBase( std::string templateFileName );
  XMLDOMBlock * createLMapHOXMLBase( std::string templateFileName );

  int addLMapHBEFDataset( XMLDOMBlock * doc, LMapRowHBEF * row, std::string templateFileName );
  int addLMapHODataset( XMLDOMBlock * doc, LMapRowHO * row, std::string templateFileName );
 
  int write( XMLDOMBlock * doc, std::string target = "stdout" );
  
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
      std::cout << "XMLProcessor::_toXMLCh(int temp): buffer overflow, the std::string will be truncated!" << std::endl;
    }
  else if ( status <0 )
    {
      std::cout << "XMLProcessor::_toXMLCh(int temp): output error" << std::endl;
    }
  XMLCh* buff = XMLString::transcode( buf );    
  return  buff;
}

inline XMLCh* XMLProcessor::_toXMLCh( double temp )
{
  char buf[100];
  int status = snprintf( buf, 100, "%.10e", temp );
  if ( status >= 100 )
    {
      std::cout << "XMLProcessor::_toXMLCh(int temp): buffer overflow, the std::string will be truncated!" << std::endl;
    }
  else if ( status <0 )
    {
      std::cout << "XMLProcessor::_toXMLCh(int temp): output error" << std::endl;
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
      std::cout << "XML  Processor::_toXMLCh(int temp): buffer overflow, the std::string is indeterminate!" << std::endl;
    }
  XMLCh* buff = XMLString::transcode( buf );    
  return  buff;
}


#endif


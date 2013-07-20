// -*- C++ -*-
//
// Package:     XMLTools
// Class  :     XMLProcessor
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Fri Sep 21 15:56:27 CEST 2007
// $Id: XMLProcessor.cc,v 1.3 2010/08/06 20:24:03 wmtan Exp $
//

// system include files
#include <vector>
#include <string>
#include <iostream>
#include <sys/types.h>
#include <pwd.h>
#include <unistd.h>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/dom/DOM.hpp>
#include <xercesc/sax/HandlerBase.hpp>
#include <xercesc/util/XMLString.hpp>
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/dom/DOMNode.hpp>
#include <xercesc/framework/StdOutFormatTarget.hpp>
#include <xercesc/framework/LocalFileFormatTarget.hpp>

// xalan-c init
//#include <xalanc/Include/PlatformDefinitions.hpp>
//#include <xalanc/XPath/XPathEvaluator.hpp>
//#include <xalanc/XalanTransformer/XalanTransformer.hpp>
//using namespace xalanc;

using namespace std;
XERCES_CPP_NAMESPACE_USE 

// user include files
#include "CalibCalorimetry/HcalTPGAlgos/interface/XMLProcessor.h"
#include "CalibCalorimetry/HcalTPGAlgos/interface/XMLDOMBlock.h"

XMLProcessor * XMLProcessor::instance = NULL;

XMLProcessor::XMLProcessor()
{
  // initializes Xerces, must be done only once
  init();
}

XMLProcessor::~XMLProcessor()
{
  // terminates Xerces
  terminate();
}

//_loaderBaseConfig::_loaderBaseConfig
XMLProcessor::loaderBaseConfig::_loaderBaseConfig()
{
  extention_table_name = "HCAL_TRIG_PRIM_LOOKUP_TABLES";
  name = "HCAL trigger primitive lookup table";
  run_mode = "no-run";
  data_set_id = "-1";
  iov_id = "1";
  iov_begin = "0";
  iov_end = "1";
  tag_id = "2";
  tag_mode = "auto";
  tag_name = "dummy tag";
  detector_name = "HCAL";
  comment_description = "empty comment";
}

XMLProcessor::DBConfig::_DBConfig()
{
  version = "test:2";
  subversion = "1";
  create_timestamp = time( NULL );
  created_by_user = getpwuid( getuid() ) -> pw_name;
}

XMLDOMBlock * XMLProcessor::createLMapHBEFXMLBase( std::string templateFileName )
{
  XMLDOMBlock * result = new XMLDOMBlock( templateFileName );
  DOMDocument * loader = result -> getDocument();
  //DOMElement * root = loader -> getDocumentElement();

  loader -> getElementsByTagName( _toXMLCh( "NAME" ) ) -> item(0) -> getFirstChild() -> setNodeValue( _toXMLCh( "HCAL LMAP for HB, HE, HF" ) );
  //DOMElement * _tag = (DOMElement *)(loader -> getElementsByTagName( _toXMLCh( "TAG" ) ) -> item(0));
  //_tag -> setAttribute( _toXMLCh("mode"), _toXMLCh("test_mode") );

  return result;
}

int XMLProcessor::addLMapHBEFDataset( XMLDOMBlock * doc, LMapRowHBEF * row, std::string templateFileName )
{
  DOMDocument * loader = doc -> getDocument();
  DOMElement * root = loader -> getDocumentElement();

  XMLDOMBlock dataSetDoc( templateFileName );
  DOMDocument * dataSet = dataSetDoc . getDocument();
  
  //Dataset
  dataSet -> getElementsByTagName( _toXMLCh( "SIDE" ) ) -> item(0) -> getFirstChild() -> setNodeValue( _toXMLCh( row -> side ) );  
  dataSet -> getElementsByTagName( _toXMLCh( "ETA" ) ) -> item(0) -> getFirstChild() -> setNodeValue( _toXMLCh( row ->  eta) );  
  dataSet -> getElementsByTagName( _toXMLCh( "PHI" ) ) -> item(0) -> getFirstChild() -> setNodeValue( _toXMLCh( row -> phi ) );  
  dataSet -> getElementsByTagName( _toXMLCh( "DELTA_PHI" ) ) -> item(0) -> getFirstChild() -> setNodeValue( _toXMLCh( row -> dphi ) );  

  dataSet -> getElementsByTagName( _toXMLCh( "DEPTH" ) ) -> item(0) -> getFirstChild() -> setNodeValue( _toXMLCh( row -> depth  ) );  
  dataSet -> getElementsByTagName( _toXMLCh( "SUBDETECTOR" ) ) -> item(0) -> getFirstChild() -> setNodeValue( _toXMLCh( row -> det ) );  
  dataSet -> getElementsByTagName( _toXMLCh( "RBX_SLOT" ) ) -> item(0) -> getFirstChild() -> setNodeValue( _toXMLCh( row -> rbx ) );  
  dataSet -> getElementsByTagName( _toXMLCh( "WEDGE" ) ) -> item(0) -> getFirstChild() -> setNodeValue( _toXMLCh( row -> wedge ) );
 
  dataSet -> getElementsByTagName( _toXMLCh( "RM_SLOT" ) ) -> item(0) -> getFirstChild() -> setNodeValue( _toXMLCh( row -> rm  ) );  
  dataSet -> getElementsByTagName( _toXMLCh( "HPD_PIXEL" ) ) -> item(0) -> getFirstChild() -> setNodeValue( _toXMLCh( row -> pixel ) );  
  dataSet -> getElementsByTagName( _toXMLCh( "QIE_SLOT" ) ) -> item(0) -> getFirstChild() -> setNodeValue( _toXMLCh( row -> qie ) );  
  dataSet -> getElementsByTagName( _toXMLCh( "ADC" ) ) -> item(0) -> getFirstChild() -> setNodeValue( _toXMLCh( row -> adc ) );  
  dataSet -> getElementsByTagName( _toXMLCh( "RM_FIBER" ) ) -> item(0) -> getFirstChild() -> setNodeValue( _toXMLCh( row -> rm_fi ) );

  dataSet -> getElementsByTagName( _toXMLCh( "FIBER_CHANNEL" ) ) -> item(0) -> getFirstChild() -> setNodeValue( _toXMLCh( row -> fi_ch));
  dataSet -> getElementsByTagName( _toXMLCh( "CRATE" ) ) -> item(0) -> getFirstChild() -> setNodeValue( _toXMLCh( row -> crate ) );  
  dataSet -> getElementsByTagName( _toXMLCh( "HTR_SLOT" ) ) -> item(0) -> getFirstChild() -> setNodeValue( _toXMLCh( row ->  htr ) );  
  dataSet -> getElementsByTagName( _toXMLCh( "HTR_FPGA" ) ) -> item(0) -> getFirstChild() -> setNodeValue( _toXMLCh( row -> fpga ) );

  dataSet -> getElementsByTagName( _toXMLCh( "HTR_FIBER" ) ) -> item(0) -> getFirstChild() -> setNodeValue( _toXMLCh( row -> htr_fi ));  
  dataSet -> getElementsByTagName( _toXMLCh( "DCC_SL" ) ) -> item(0) -> getFirstChild() -> setNodeValue( _toXMLCh( row -> dcc_sl ) );  
  dataSet -> getElementsByTagName( _toXMLCh( "SPIGOT" ) ) -> item(0) -> getFirstChild() -> setNodeValue( _toXMLCh( row ->  spigo ) );  
  dataSet -> getElementsByTagName( _toXMLCh( "DCC_SLOT" ) ) -> item(0) -> getFirstChild() -> setNodeValue( _toXMLCh( row -> dcc ) );  
  dataSet -> getElementsByTagName( _toXMLCh( "SLB_SITE" ) ) -> item(0) -> getFirstChild() -> setNodeValue( _toXMLCh( row -> slb ) );  
  dataSet -> getElementsByTagName( _toXMLCh( "SLB_CHANNEL" ) ) -> item(0) -> getFirstChild() -> setNodeValue( _toXMLCh( row -> slbin )); 
  dataSet -> getElementsByTagName( _toXMLCh( "SLB_CHANNEL2" ) ) -> item(0) -> getFirstChild() -> setNodeValue( _toXMLCh( row -> slbin2));

  dataSet -> getElementsByTagName( _toXMLCh( "SLB_CABLE" ) ) -> item(0) -> getFirstChild() -> setNodeValue( _toXMLCh( row -> slnam ) );  
  dataSet -> getElementsByTagName( _toXMLCh( "RCT_CRATE" ) ) -> item(0) -> getFirstChild() -> setNodeValue( _toXMLCh( row -> rctcra ) ); 
  dataSet -> getElementsByTagName( _toXMLCh( "RCT_CARD" ) ) -> item(0) -> getFirstChild() -> setNodeValue( _toXMLCh( row -> rctcar ) );
  dataSet -> getElementsByTagName( _toXMLCh( "RCT_CONNECTOR" ) ) -> item(0) -> getFirstChild() -> setNodeValue( _toXMLCh( row ->rctcon));
  dataSet -> getElementsByTagName( _toXMLCh( "RCT_NAME" ) ) -> item(0) -> getFirstChild() -> setNodeValue( _toXMLCh( row -> rctnam ) );  
  dataSet -> getElementsByTagName( _toXMLCh( "FED_ID" ) ) -> item(0) -> getFirstChild() -> setNodeValue( _toXMLCh( row -> fedid ) );
  
      
  // copy the <data_set> node into the final XML
  DOMNode * cloneDataSet = loader -> importNode( dataSet -> getDocumentElement(), true );
  root -> appendChild( cloneDataSet );

  return 0;
}

XMLDOMBlock * XMLProcessor::createLMapHOXMLBase( std::string templateFileName )
{
  XMLDOMBlock * result = new XMLDOMBlock( templateFileName );
  DOMDocument * loader = result -> getDocument();
  //DOMElement * root = loader -> getDocumentElement();

  loader -> getElementsByTagName( _toXMLCh( "NAME" ) ) -> item(0) -> getFirstChild() -> setNodeValue( _toXMLCh( "HCAL LMAP for HO" ) );
  //DOMElement * _tag = (DOMElement *)(loader -> getElementsByTagName( _toXMLCh( "TAG" ) ) -> item(0));
  //_tag -> setAttribute( _toXMLCh("mode"), _toXMLCh("test_mode") );

  return result;
}

int XMLProcessor::addLMapHODataset( XMLDOMBlock * doc, LMapRowHO * row, std::string templateFileName )
{
  DOMDocument * loader = doc -> getDocument();
  DOMElement * root = loader -> getDocumentElement();

  XMLDOMBlock dataSetDoc( templateFileName );
  DOMDocument * dataSet = dataSetDoc . getDocument();
  
  //Dataset
  dataSet -> getElementsByTagName( _toXMLCh( "SIDE" ) ) -> item(0) -> getFirstChild() -> setNodeValue( _toXMLCh( row -> sideO ) );  
  dataSet -> getElementsByTagName( _toXMLCh( "ETA" ) ) -> item(0) -> getFirstChild() -> setNodeValue( _toXMLCh( row ->  etaO ) );  
  dataSet -> getElementsByTagName( _toXMLCh( "PHI" ) ) -> item(0) -> getFirstChild() -> setNodeValue( _toXMLCh( row -> phiO ) );  
  dataSet -> getElementsByTagName( _toXMLCh( "DELTA_PHI" ) ) -> item(0) -> getFirstChild() -> setNodeValue( _toXMLCh( row -> dphiO ) );
  
  dataSet -> getElementsByTagName( _toXMLCh( "DEPTH" ) ) -> item(0) -> getFirstChild() -> setNodeValue( _toXMLCh( row -> depthO  ) );  
  dataSet -> getElementsByTagName( _toXMLCh( "SUBDETECTOR" ) ) -> item(0) -> getFirstChild() -> setNodeValue( _toXMLCh( row -> detO ) ); 
  dataSet -> getElementsByTagName( _toXMLCh( "RBX_SLOT" ) ) -> item(0) -> getFirstChild() -> setNodeValue( _toXMLCh( row -> rbxO ) );  
  dataSet -> getElementsByTagName( _toXMLCh( "SECTOR" ) ) -> item(0) -> getFirstChild() -> setNodeValue( _toXMLCh( row -> sectorO ) );
 
  dataSet -> getElementsByTagName( _toXMLCh( "RM_SLOT" ) ) -> item(0) -> getFirstChild() -> setNodeValue( _toXMLCh( row -> rmO  ) );  
  dataSet -> getElementsByTagName( _toXMLCh( "HPD_PIXEL" ) ) -> item(0) -> getFirstChild() -> setNodeValue( _toXMLCh( row -> pixelO ) ); 
  dataSet -> getElementsByTagName( _toXMLCh( "QIE_SLOT" ) ) -> item(0) -> getFirstChild() -> setNodeValue( _toXMLCh( row -> qieO ) );  
  dataSet -> getElementsByTagName( _toXMLCh( "ADC" ) ) -> item(0) -> getFirstChild() -> setNodeValue( _toXMLCh( row -> adcO ) );  
  dataSet -> getElementsByTagName( _toXMLCh( "RM_FIBER" ) ) -> item(0) -> getFirstChild() -> setNodeValue( _toXMLCh( row -> rm_fiO ) );  
  dataSet -> getElementsByTagName( _toXMLCh( "FIBER_CHANNEL" ) ) -> item(0) -> getFirstChild() -> setNodeValue( _toXMLCh( row ->fi_chO));

  dataSet -> getElementsByTagName( _toXMLCh( "LETTER_CODE" ) ) -> item(0) -> getFirstChild() -> setNodeValue( _toXMLCh(row ->let_codeO));
  dataSet -> getElementsByTagName( _toXMLCh( "CRATE" ) ) -> item(0) -> getFirstChild() -> setNodeValue( _toXMLCh( row -> crateO ) );  
  dataSet -> getElementsByTagName( _toXMLCh( "HTR_SLOT" ) ) -> item(0) -> getFirstChild() -> setNodeValue( _toXMLCh( row ->  htrO ) );  
  dataSet -> getElementsByTagName( _toXMLCh( "HTR_FPGA" ) ) -> item(0) -> getFirstChild() -> setNodeValue( _toXMLCh( row -> fpgaO ) );

  dataSet -> getElementsByTagName( _toXMLCh( "HTR_FIBER" ) ) -> item(0) -> getFirstChild() -> setNodeValue( _toXMLCh( row -> htr_fiO ) );
  dataSet -> getElementsByTagName( _toXMLCh( "DCC_SL" ) ) -> item(0) -> getFirstChild() -> setNodeValue( _toXMLCh( row -> dcc_slO ) );  
  dataSet -> getElementsByTagName( _toXMLCh( "SPIGOT" ) ) -> item(0) -> getFirstChild() -> setNodeValue( _toXMLCh( row ->  spigoO ) );  
  dataSet -> getElementsByTagName( _toXMLCh( "DCC_SLOT" ) ) -> item(0) -> getFirstChild() -> setNodeValue( _toXMLCh( row -> dccO ) );  
  dataSet -> getElementsByTagName( _toXMLCh( "FED_ID" ) ) -> item(0) -> getFirstChild() -> setNodeValue( _toXMLCh( row -> fedidO ) );
  
      
  // copy the <data_set> node into the final XML
  DOMNode * cloneDataSet = loader -> importNode( dataSet -> getDocumentElement(), true );
  root -> appendChild( cloneDataSet );

  return 0;
}

int XMLProcessor::write( XMLDOMBlock * doc, std::string target )
{
  DOMDocument * loader = doc -> getDocument();
  //DOMElement * root = loader -> getDocumentElement();

  XMLCh * _t;
  _t = serializeDOM( loader, target );
  delete _t;

  return 0;
}

int XMLProcessor::test( void )
{
  //if ( init() != 0 ) return 1;

  XMLDOMBlock dataBlock( "HCAL_TRIG_PRIM_LOOKUP_TABLE.datablock.template" );

  DOMDocument * dataBlockDocument = dataBlock . getDocument();

  std::cout << "===> Tag length: " << dataBlockDocument -> getElementsByTagName( _toXMLCh( "CREATED_BY_USER" ) ) -> getLength() << std::endl;
  std::cout << "===> Tag name: " << XMLString::transcode( dataBlockDocument -> getElementsByTagName( _toXMLCh( "CREATED_BY_USER" ) ) -> item(0) -> getNodeName() ) << std::endl;
  dataBlockDocument -> getElementsByTagName( _toXMLCh( "CREATED_BY_USER" ) ) -> item(0) -> getFirstChild() -> setNodeValue( _toXMLCh( "kukarzev test" ) );

  XMLCh * _t;
  _t = serializeDOM( dataBlockDocument );
  delete _t;

  //terminate();

  return 0;
}

XMLCh * XMLProcessor::serializeDOM(DOMNode* node, std::string target)
{
  XMLCh tempStr[100];
  XMLString::transcode("LS", tempStr, 99);
  DOMImplementation *impl = DOMImplementationRegistry::getDOMImplementation(tempStr);
  DOMWriter* theSerializer = ((DOMImplementationLS*)impl)->createDOMWriter();
  
  if (theSerializer->canSetFeature(XMLUni::fgDOMWRTDiscardDefaultContent, true))
    theSerializer->setFeature(XMLUni::fgDOMWRTDiscardDefaultContent, true);
  
  if (theSerializer->canSetFeature(XMLUni::fgDOMWRTFormatPrettyPrint, true))
    theSerializer->setFeature(XMLUni::fgDOMWRTFormatPrettyPrint, true);
    
  XMLFormatTarget * myFormTarget = 0;
  XMLCh * _string = 0;
  if ( target == "stdout" || target == "string" )
    {
      myFormTarget = new StdOutFormatTarget();
    }
  //else if ( target == "memory" )
  //  {
  //    myFormTarget = new MemBufFormatTarget();
  //  }
  else
    {
      myFormTarget = new LocalFileFormatTarget( _toXMLCh( target ) );
    }
  
  try {
    if ( target == "string" ){
      _string = theSerializer->writeToString( *node );
    }
    else{
      theSerializer->writeNode(myFormTarget, *node);
    }
  }
  catch (const XMLException& toCatch) {
    char* message = XMLString::transcode(toCatch.getMessage());
    std::cout << "Exception message is: \n"
	 << message << "\n";
    XMLString::release(&message);
    return 0;
  }
  catch (const DOMException& toCatch) {
    char* message = XMLString::transcode(toCatch.msg);
    std::cout << "Exception message is: \n"
	 << message << "\n";
    XMLString::release(&message);
    return NULL;
  }
  catch (...) {
    std::cout << "Unexpected Exception \n" ;
    return NULL;
  }
    
  theSerializer->release();
  if ( myFormTarget ) delete myFormTarget;
  return _string;
}

int XMLProcessor::init( void )
{
  std::cerr << "Intializing Xerces-c...";
  try {
    XMLPlatformUtils::Initialize();
    //
    //_____ following removed as a xalan-c component_____________________
    //
    //XPathEvaluator::initialize();
  }
  catch (const XMLException& toCatch) {
    std::cout << " FAILED! Exiting..." << std::endl;
    return 1;
  }
  std::cerr << " done" << std::endl;

  return 0;
}

int XMLProcessor::terminate( void )
{
  //
  //_____ following removed as a xalan-c component_____________________
  //
  //std::cout << "Terminating Xalan-c...";
  //XPathEvaluator::terminate();
  //std::cout << " done" << std::endl;

  std::cout << "Terminating Xerces-c...";
  XMLPlatformUtils::Terminate();
  std::cout << " done" << std::endl;


  // Other terminations and cleanup.

  return 0;
}



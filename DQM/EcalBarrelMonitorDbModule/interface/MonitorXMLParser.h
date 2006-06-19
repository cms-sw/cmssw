// $Id: MonitorXMLParser.h,v 1.2 2006/06/16 15:41:12 benigno Exp $

/*!
  \file MonitorXMLParser.h
  \brief monitor db xml elements parsing tool
  \author B. Gobbo 
  \version $Revision: 1.2 $
  \date $Date: 2006/06/16 15:41:12 $
*/

#ifndef MonitorXMLParser_h
#define MonitorXMLParser_h

#include <string>
#include <vector>
#include <map>
#include <iostream>

#include <xercesc/dom/DOM.hpp>
#include <xercesc/dom/DOMElement.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>

// - - - - - - - - - - - - - - - - - - - -

enum { ERROR_ARGS = 1 ,
       ERROR_XERCES_INIT ,
       ERROR_PARSE ,
       ERROR_EMPTY_DOCUMENT
};

// - - - - - - - - - - - - - - - - - - - -

typedef struct { std::string type; std::string title; int xbins; double xfrom; double xto; 
  int ybins; double yfrom; double yto; int zbins; double zfrom; double zto; 
std::multimap< std::string, std::string > queries; } DB_ME;

// - - - - - - - - - - - - - - - - - - - -

class TagNames {

public:
  XMLCh* TAG_DBE;
  XMLCh* TAG_ME;
  XMLCh* TAG_1D;
  XMLCh* TAG_2D;
  XMLCh* TAG_TPROFILE;
  XMLCh* TAG_TPROFILE2D;
  XMLCh* TAG_QUERY;

  XMLCh* ATTR_TITLE;
  XMLCh* ATTR_XBINS;
  XMLCh* ATTR_XFROM;
  XMLCh* ATTR_XTO;
  XMLCh* ATTR_YBINS;
  XMLCh* ATTR_YFROM;
  XMLCh* ATTR_YTO;
  XMLCh* ATTR_ZBINS;
  XMLCh* ATTR_ZFROM;
  XMLCh* ATTR_ZTO;
  XMLCh* ATTR_NAME;
  XMLCh* ATTR_ARG;
  

  TagNames() :
    TAG_DBE( xercesc::XMLString::transcode( "dbelements" ) ),
    TAG_ME( xercesc::XMLString::transcode( "me" ) ),
    TAG_1D( xercesc::XMLString::transcode( "th1d" ) ),
    TAG_2D( xercesc::XMLString::transcode( "th2d" ) ),
    TAG_TPROFILE( xercesc::XMLString::transcode( "tprofile" ) ),
    TAG_TPROFILE2D( xercesc::XMLString::transcode( "tprofile2d" ) ),
    TAG_QUERY( xercesc::XMLString::transcode( "query" ) ),

    ATTR_TITLE( xercesc::XMLString::transcode( "title" ) ),
    ATTR_XBINS( xercesc::XMLString::transcode( "xbins" ) ),   
    ATTR_XFROM( xercesc::XMLString::transcode( "xfrom" ) ),
    ATTR_XTO( xercesc::XMLString::transcode( "xto" ) ),
    ATTR_YBINS( xercesc::XMLString::transcode( "ybins" ) ),
    ATTR_YFROM( xercesc::XMLString::transcode( "yfrom" ) ),
    ATTR_YTO( xercesc::XMLString::transcode( "yto" ) ),
    ATTR_ZBINS( xercesc::XMLString::transcode( "ybins" ) ),
    ATTR_ZFROM( xercesc::XMLString::transcode( "yfrom" ) ),
    ATTR_ZTO( xercesc::XMLString::transcode( "yto" ) ),
    ATTR_NAME( xercesc::XMLString::transcode( "name" ) ),
    ATTR_ARG( xercesc::XMLString::transcode( "arg" ) )  {

    return ;

  }


  ~TagNames() throw(){
    
    try{
      xercesc::XMLString::release( &TAG_DBE ) ;
      xercesc::XMLString::release( &TAG_ME ) ;
      xercesc::XMLString::release( &TAG_1D ) ;
      xercesc::XMLString::release( &TAG_2D ) ;
      xercesc::XMLString::release( &TAG_TPROFILE ) ;
      xercesc::XMLString::release( &TAG_TPROFILE2D ) ;
      xercesc::XMLString::release( &TAG_QUERY ) ;
      
      xercesc::XMLString::release( &ATTR_TITLE ) ;
      xercesc::XMLString::release( &ATTR_XFROM ) ;
      xercesc::XMLString::release( &ATTR_XTO ) ;
      xercesc::XMLString::release( &ATTR_XBINS ) ;
      xercesc::XMLString::release( &ATTR_YFROM ) ;
      xercesc::XMLString::release( &ATTR_YTO ) ;
      xercesc::XMLString::release( &ATTR_YBINS ) ;
      xercesc::XMLString::release( &ATTR_NAME ) ;
      xercesc::XMLString::release( &ATTR_ARG ) ;

    }catch( ... ){
      std::cerr << "Unknown exception encountered in TagNames dtor" << std::endl ;
    }
	
  } 

}; // class TagNames


// - - - - - - - - - - - - - - - - - - - -


class MonitorXMLParser {

private:
  
  std::vector< DB_ME >      DBMonitoringElements_;
  std::string               xmlFile_;
  xercesc::XercesDOMParser* parser_;
  TagNames*                 tags_;
  void handleElement( xercesc::DOMElement* element );

public:

  MonitorXMLParser( const std::string& fromFile );

  ~MonitorXMLParser() throw();

  const std::vector< DB_ME > & getDB_ME( void ) const { return (DBMonitoringElements_ ); }

  void load() throw( std::runtime_error ); 

}; // class MonitorXMLParser

#endif // MonitorXMLParser_h

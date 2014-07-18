
/*!
  \file MonitorXMLParser.cc
  \brief monitor db xml elements parsing tool
  \author B. Gobbo
*/

#include "FWCore/Concurrency/interface/Xerces.h"

#include <xercesc/dom/DOMDocument.hpp>
#include <xercesc/dom/DOMNodeList.hpp>

#include <sstream>
#include <stdexcept>

#include "../interface/MonitorXMLParser.h"

MonitorXMLParser::MonitorXMLParser( const std::string& fromFile ) {

  try{

    cms::concurrency::xercesInitialize();

  }catch( xercesc::XMLException& e ){

    char* message = xercesc::XMLString::transcode( e.getMessage() ) ;

    std::cerr << "XML toolkit initialization error: " << message << std::endl;

    xercesc::XMLString::release( &message );

    exit( ERROR_XERCES_INIT );

  }

  xmlFile_ = fromFile;
  parser_  = new xercesc::XercesDOMParser();
  tags_    = new TagNames();

}

// - - - - - - - - - - - - - - - - -

MonitorXMLParser::~MonitorXMLParser() throw() {

  try{
    cms::concurrency::xercesTerminate();
  } catch ( xercesc::XMLException& e ){
    char* message = xercesc::XMLString::transcode( e.getMessage() );
    std::cerr << "XML toolkit teardown error: " << message << std::endl;
    xercesc::XMLString::release( &message ) ;
  }

  delete parser_;
  delete tags_;

}



// - - - - - - - - - - - - - - - - -

void MonitorXMLParser::handleElement( xercesc::DOMElement* element ){

  if( xercesc::XMLString::equals( tags_->TAG_ME, element->getTagName() ) ) {

    char* c;
    std::stringstream s;
    DB_ME me;
    bool meok;

    meok = false;

    xercesc::DOMNodeList* d1Nodes = element->getElementsByTagName( tags_->TAG_1D );
    const XMLSize_t d1Count = d1Nodes->getLength();

    for( XMLSize_t d1Index = 0; d1Index < d1Count; ++d1Index ){

      xercesc::DOMNode* d1Node = d1Nodes->item( d1Index ) ;

      xercesc::DOMElement* d1Element = dynamic_cast< xercesc::DOMElement* >( d1Node ) ;

      const XMLCh* d1titleXMLCh = d1Element->getAttribute( tags_->ATTR_TITLE ) ;
      c = xercesc::XMLString::transcode( d1titleXMLCh );
      me.type = "th1d";
      me.title = c;
      meok = true;
      xercesc::XMLString::release( &c );

      const XMLCh* d1xbinsXMLCh = d1Element->getAttribute( tags_->ATTR_XBINS ) ;
      c = xercesc::XMLString::transcode( d1xbinsXMLCh );
      s.clear(); s.str( c );
      s >> me.xbins;
      xercesc::XMLString::release( &c );

      const XMLCh* d1xfromXMLCh = d1Element->getAttribute( tags_->ATTR_XFROM ) ;
      c = xercesc::XMLString::transcode( d1xfromXMLCh );
      s.clear(); s.str( c );
      s >> me.xfrom;
      xercesc::XMLString::release( &c );

      const XMLCh* d1xtoXMLCh = d1Element->getAttribute( tags_->ATTR_XTO ) ;
      c = xercesc::XMLString::transcode( d1xtoXMLCh );
      s.clear(); s.str( c );
      s >> me.xto;
      xercesc::XMLString::release( &c );

      const XMLCh* d1ncycleXMLCh = d1Element->getAttribute( tags_->ATTR_NCYCLE ) ;
      c = xercesc::XMLString::transcode( d1ncycleXMLCh );
      s.clear(); s.str( c );
      s >> me.ncycle;
      xercesc::XMLString::release( &c );

      const XMLCh* d1loopXMLCh = d1Element->getAttribute( tags_->ATTR_LOOP ) ;
      c = xercesc::XMLString::transcode( d1loopXMLCh );
      s.clear(); s.str( c );
      s >> me.loop;
      xercesc::XMLString::release( &c );

      me.ybins = 0;
      me.yfrom = 0.0;
      me.yto = 0.0;
      me.zbins = 0;
      me.zfrom = 0.0;
      me.zto = 0.0;

    }

    xercesc::DOMNodeList* d2Nodes = element->getElementsByTagName( tags_->TAG_2D );
    const XMLSize_t d2Count = d2Nodes->getLength();

    for( XMLSize_t d2Index = 0; d2Index < d2Count; ++d2Index ){

      xercesc::DOMNode* d2Node = d2Nodes->item( d2Index ) ;

      xercesc::DOMElement* d2Element = dynamic_cast< xercesc::DOMElement* >( d2Node ) ;

      const XMLCh* d2titleXMLCh = d2Element->getAttribute( tags_->ATTR_TITLE ) ;
      c = xercesc::XMLString::transcode( d2titleXMLCh );
      me.type = "th2d";
      me.title = c;
      meok = true;
      xercesc::XMLString::release( &c );

      const XMLCh* d2xbinsXMLCh = d2Element->getAttribute( tags_->ATTR_XBINS ) ;
      c = xercesc::XMLString::transcode( d2xbinsXMLCh );
      s.clear(); s.str( c );
      s >> me.xbins;
      xercesc::XMLString::release( &c );

      const XMLCh* d2xfromXMLCh = d2Element->getAttribute( tags_->ATTR_XFROM ) ;
      c = xercesc::XMLString::transcode( d2xfromXMLCh );
      s.clear(); s.str( c );
      s >> me.xfrom;
      xercesc::XMLString::release( &c );

      const XMLCh* d2xtoXMLCh = d2Element->getAttribute( tags_->ATTR_XTO ) ;
      c = xercesc::XMLString::transcode( d2xtoXMLCh );
      s.clear(); s.str( c );
      s >> me.xto;
      xercesc::XMLString::release( &c );

      const XMLCh* d2ybinsXMLCh = d2Element->getAttribute( tags_->ATTR_YBINS ) ;
      c = xercesc::XMLString::transcode( d2ybinsXMLCh );
      s.clear(); s.str( c );
      s >> me.ybins;
      xercesc::XMLString::release( &c );

      const XMLCh* d2yfromXMLCh = d2Element->getAttribute( tags_->ATTR_YFROM ) ;
      c = xercesc::XMLString::transcode( d2yfromXMLCh );
      s.clear(); s.str( c );
      s >> me.yfrom;
      xercesc::XMLString::release( &c );

      const XMLCh* d2ytoXMLCh = d2Element->getAttribute( tags_->ATTR_YTO ) ;
      c = xercesc::XMLString::transcode( d2ytoXMLCh );
      s.clear(); s.str( c );
      s >> me.yto;
      xercesc::XMLString::release( &c );

      const XMLCh* d2ncycleXMLCh = d2Element->getAttribute( tags_->ATTR_NCYCLE ) ;
      c = xercesc::XMLString::transcode( d2ncycleXMLCh );
      s.clear(); s.str( c );
      s >> me.ncycle;
      xercesc::XMLString::release( &c );

      const XMLCh* d2loopXMLCh = d2Element->getAttribute( tags_->ATTR_LOOP ) ;
      c = xercesc::XMLString::transcode( d2loopXMLCh );
      s.clear(); s.str( c );
      s >> me.loop;
      xercesc::XMLString::release( &c );

      me.zbins = 0;
      me.zfrom = 0.0;
      me.zto = 0.0;

    }

    xercesc::DOMNodeList* tpNodes = element->getElementsByTagName( tags_->TAG_TPROFILE );
    const XMLSize_t tpCount = tpNodes->getLength();

    for( XMLSize_t tpIndex = 0; tpIndex < tpCount; ++tpIndex ){

      xercesc::DOMNode* tpNode = tpNodes->item( tpIndex ) ;

      xercesc::DOMElement* tpElement = dynamic_cast< xercesc::DOMElement* >( tpNode ) ;

      const XMLCh* tptitleXMLCh = tpElement->getAttribute( tags_->ATTR_TITLE ) ;
      c = xercesc::XMLString::transcode( tptitleXMLCh );
      me.type = "tprofile";
      me.title = c;
      meok = true;
      xercesc::XMLString::release( &c );

      const XMLCh* tpxbinsXMLCh = tpElement->getAttribute( tags_->ATTR_XBINS ) ;
      c = xercesc::XMLString::transcode( tpxbinsXMLCh );
      s.clear(); s.str( c );
      s >> me.xbins;
      xercesc::XMLString::release( &c );

      const XMLCh* tpxfromXMLCh = tpElement->getAttribute( tags_->ATTR_XFROM ) ;
      c = xercesc::XMLString::transcode( tpxfromXMLCh );
      s.clear(); s.str( c );
      s >> me.xfrom;
      xercesc::XMLString::release( &c );

      const XMLCh* tpxtoXMLCh = tpElement->getAttribute( tags_->ATTR_XTO ) ;
      c = xercesc::XMLString::transcode( tpxtoXMLCh );
      s.clear(); s.str( c );
      s >> me.xto;
      xercesc::XMLString::release( &c );

      const XMLCh* tpybinsXMLCh = tpElement->getAttribute( tags_->ATTR_YBINS ) ;
      c = xercesc::XMLString::transcode( tpybinsXMLCh );
      s.clear(); s.str( c );
      s >> me.ybins;
      xercesc::XMLString::release( &c );

      const XMLCh* tpyfromXMLCh = tpElement->getAttribute( tags_->ATTR_YFROM ) ;
      c = xercesc::XMLString::transcode( tpyfromXMLCh );
      s.clear(); s.str( c );
      s >> me.yfrom;
      xercesc::XMLString::release( &c );

      const XMLCh* tpytoXMLCh = tpElement->getAttribute( tags_->ATTR_YTO ) ;
      c = xercesc::XMLString::transcode( tpytoXMLCh );
      s.clear(); s.str( c );
      s >> me.yto;
      xercesc::XMLString::release( &c );

      const XMLCh* tpncycleXMLCh = tpElement->getAttribute( tags_->ATTR_NCYCLE ) ;
      c = xercesc::XMLString::transcode( tpncycleXMLCh );
      s.clear(); s.str( c );
      s >> me.ncycle;
      xercesc::XMLString::release( &c );

      const XMLCh* tploopXMLCh = tpElement->getAttribute( tags_->ATTR_LOOP ) ;
      c = xercesc::XMLString::transcode( tploopXMLCh );
      s.clear(); s.str( c );
      s >> me.loop;
      xercesc::XMLString::release( &c );

      me.zbins = 0;
      me.zfrom = 0.0;
      me.zto = 0.0;

    }

    xercesc::DOMNodeList* tp2dNodes = element->getElementsByTagName( tags_->TAG_TPROFILE2D );
    const XMLSize_t tp2dCount = tp2dNodes->getLength();

    for( XMLSize_t tp2dIndex = 0; tp2dIndex < tp2dCount; ++tp2dIndex ){

      xercesc::DOMNode* tp2dNode = tp2dNodes->item( tp2dIndex ) ;

      xercesc::DOMElement* tp2dElement = dynamic_cast< xercesc::DOMElement* >( tp2dNode ) ;

      const XMLCh* tp2dtitleXMLCh = tp2dElement->getAttribute( tags_->ATTR_TITLE ) ;
      c = xercesc::XMLString::transcode( tp2dtitleXMLCh );
      me.type = "tprofile2d";
      me.title = c;
      meok = true;
      xercesc::XMLString::release( &c );

      const XMLCh* tp2dxbinsXMLCh = tp2dElement->getAttribute( tags_->ATTR_XBINS ) ;
      c = xercesc::XMLString::transcode( tp2dxbinsXMLCh );
      s.clear(); s.str( c );
      s >> me.xbins;
      xercesc::XMLString::release( &c );

      const XMLCh* tp2dxfromXMLCh = tp2dElement->getAttribute( tags_->ATTR_XFROM ) ;
      c = xercesc::XMLString::transcode( tp2dxfromXMLCh );
      s.clear(); s.str( c );
      s >> me.xfrom;
      xercesc::XMLString::release( &c );

      const XMLCh* tp2dxtoXMLCh = tp2dElement->getAttribute( tags_->ATTR_XTO ) ;
      c = xercesc::XMLString::transcode( tp2dxtoXMLCh );
      s.clear(); s.str( c );
      s >> me.xto;
      xercesc::XMLString::release( &c );

      const XMLCh* tp2dybinsXMLCh = tp2dElement->getAttribute( tags_->ATTR_YBINS ) ;
      c = xercesc::XMLString::transcode( tp2dybinsXMLCh );
      s.clear(); s.str( c );
      s >> me.ybins;
      xercesc::XMLString::release( &c );

      const XMLCh* tp2dyfromXMLCh = tp2dElement->getAttribute( tags_->ATTR_YFROM ) ;
      c = xercesc::XMLString::transcode( tp2dyfromXMLCh );
      s.clear(); s.str( c );
      s >> me.yfrom;
      xercesc::XMLString::release( &c );

      const XMLCh* tp2dytoXMLCh = tp2dElement->getAttribute( tags_->ATTR_YTO ) ;
      c = xercesc::XMLString::transcode( tp2dytoXMLCh );
      s.clear(); s.str( c );
      s >> me.yto;
      xercesc::XMLString::release( &c );

      const XMLCh* tp2dzbinsXMLCh = tp2dElement->getAttribute( tags_->ATTR_ZBINS ) ;
      c = xercesc::XMLString::transcode( tp2dzbinsXMLCh );
      s.clear(); s.str( c );
      s >> me.zbins;
      xercesc::XMLString::release( &c );

      const XMLCh* tp2dzfromXMLCh = tp2dElement->getAttribute( tags_->ATTR_ZFROM ) ;
      c = xercesc::XMLString::transcode( tp2dzfromXMLCh );
      s.clear(); s.str( c );
      s >> me.zfrom;
      xercesc::XMLString::release( &c );

      const XMLCh* tp2dztoXMLCh = tp2dElement->getAttribute( tags_->ATTR_ZTO ) ;
      c = xercesc::XMLString::transcode( tp2dztoXMLCh );
      s.clear(); s.str( c );
      s >> me.zto;
      xercesc::XMLString::release( &c );

      const XMLCh* tp2dncycleXMLCh = tp2dElement->getAttribute( tags_->ATTR_NCYCLE ) ;
      c = xercesc::XMLString::transcode( tp2dncycleXMLCh );
      s.clear(); s.str( c );
      s >> me.ncycle;
      xercesc::XMLString::release( &c );

      const XMLCh* tp2dloopXMLCh = tp2dElement->getAttribute( tags_->ATTR_LOOP ) ;
      c = xercesc::XMLString::transcode( tp2dloopXMLCh );
      s.clear(); s.str( c );
      s >> me.loop;
      xercesc::XMLString::release( &c );

    }


    xercesc::DOMNodeList* qNodes = element->getElementsByTagName( tags_->TAG_QUERY );
    const XMLSize_t qCount = qNodes->getLength();

    for( XMLSize_t qIndex = 0; qIndex < qCount; ++qIndex ){

      xercesc::DOMNode* qNode = qNodes->item( qIndex ) ;

      xercesc::DOMElement* qElement = dynamic_cast< xercesc::DOMElement* >( qNode ) ;

      const XMLCh* nameXMLCh = qElement->getAttribute( tags_->ATTR_NAME ) ;
      c = xercesc::XMLString::transcode( nameXMLCh );

      const XMLCh* argXMLCh = qElement->getAttribute( tags_->ATTR_ARG ) ;
      char* d = xercesc::XMLString::transcode( argXMLCh );

      const XMLCh* aliasXMLCh = qElement->getAttribute( tags_->ATTR_ALIAS ) ;
      char* e = xercesc::XMLString::transcode( aliasXMLCh );

      DbQuery tmpQuery;
      tmpQuery.query = c;
      tmpQuery.arg = d;
      tmpQuery.alias = e;

      me.queries.push_back( tmpQuery );

      xercesc::XMLString::release( &c );
      xercesc::XMLString::release( &d );
      xercesc::XMLString::release( &e );


    }

    if( meok ) DBMonitoringElements_.push_back( me );

  }
} // handleElement()


// - - - - - - - - - - - - - - - - - - -

void MonitorXMLParser::load() throw( std::runtime_error ) {

  parser_->setValidationScheme( xercesc::XercesDOMParser::Val_Never );
  parser_->setDoNamespaces( false );
  parser_->setDoSchema( false );
  parser_->setLoadExternalDTD( false );

  try{

    parser_->parse( xmlFile_.c_str() );

    xercesc::DOMDocument* xmlDoc = parser_->getDocument();

    xercesc::DOMElement* dbe = xmlDoc->getDocumentElement();

    if( NULL == dbe ){
      throw( std::runtime_error( "empty XML document" ) ) ;
    }

    if( xercesc::XMLString::equals( tags_->TAG_DBE, dbe->getTagName() ) ) {

      xercesc::DOMNodeList* children = dbe->getChildNodes();
      const XMLSize_t nodeCount = children->getLength();

      for( XMLSize_t ix = 0 ; ix < nodeCount ; ++ix ){
	xercesc::DOMNode* currentNode = children->item( ix );
	if( NULL == currentNode ){
	  // null node...
	  continue;
	}

	if( xercesc::DOMNode::ELEMENT_NODE != currentNode->getNodeType() ){
	  continue;
	}

	xercesc::DOMElement* currentElement = dynamic_cast< xercesc::DOMElement* >( currentNode );

	handleElement( currentElement );

      }
    }

  }catch( xercesc::XMLException& e ){

    char* message = xercesc::XMLString::transcode( e.getMessage() );

    std::ostringstream buf ;
    buf << "Error parsing file: " << message << std::endl;

    xercesc::XMLString::release( &message );

    throw( std::runtime_error( buf.str() ) );

  }catch( const xercesc::DOMException& e ){

    char* message = xercesc::XMLString::transcode( e.getMessage() );

    std::ostringstream buf;
    buf << "Encountered DOM Exception: " << message << std::endl;

    xercesc::XMLString::release( &message );

    throw( std::runtime_error( buf.str() ) );

  }

  return;

} // load()


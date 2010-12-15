// -*- c++ -*-
// $Id: XHTMLMaker.h,v 1.8 2010/03/21 08:51:03 elmer Exp $
/// @file: XHTMLMaker.h 

#ifndef XHTMLMAKER_H
#define XHTMLMAKER_H

#include <xercesc/dom/DOM.hpp>
#include <xercesc/dom/DOMDocument.hpp>
#include <xercesc/dom/DOMWriter.hpp>
#include <xercesc/util/XMLString.hpp>

#include <map>
#include <string>
#include <iostream>
#include <stdint.h>

/**
   Helper class to build XHTML pages

   $Author: elmer $
   $Revision: 1.8 $
   $Date: 2010/03/21 08:51:03 $
*/

class XHTMLMaker
{

public:

  /**
     Typedefs
  */
  typedef xercesc::DOMElement Node;
  typedef std::map<std::string,std::string> AttrMap;

  /**
     Constructor
  */
  XHTMLMaker();

  /**
     Destructor
  */
  ~XHTMLMaker();

  /**
     Initialize page and return body element
  */
  Node* start( const std::string& title );

  /**
     Useful for css and javascript
  */
  Node* getHead() const { return _head; }

  /**
     Add child
  */
  Node* addNode( const std::string& name,
                 Node* parent,
                 const AttrMap& attrs );

  /**
     Add child to top level
  */
  Node* addNode( const std::string& name, const AttrMap& attrs )
  {
    return addNode( name, (Node*)0, attrs );
  }

  /**
     Add child without attributes
  */
  Node* addNode( const std::string& name, Node* parent )
  {
    AttrMap empty;
    return addNode( name, parent, empty );
  }

  /**
     Add child to top without attributes
  */
  Node* addNode( const std::string& name )
  {
    return addNode( name, (Node*)0 );
  }

  /**
     Add text
  */
  void addText( Node* parent, const std::string& );

  /**
     Add an int32_t
  */
  void addInt( Node* parent, const int32_t& );

  /**
     Add an uint32_t
  */
  void addInt( Node* parent, const uint32_t& );

  /**
     Add an int64_t
  */
  void addInt( Node* parent, const int64_t& );

  /**
     Add an uint64_t
  */
  void addInt( Node* parent, const uint64_t& );

  /**
     Add a double
  */
  void addDouble( Node* parent, const double& value, const unsigned int& precision = 2 );

  /**
     Dump the page to stdout
  */
  void out();

  /**
     Dump the page to a local file
  */
  void out( const std::string& dest );

  /**
     Dump the page to a string
  */
  void out( std::string& dest );

  /**
     Dump the page into an output stream
  */
  void out( std::ostream& dest );

private:

  xercesc::DOMDocument* _doc;
  xercesc::DOMWriter* _writer;
  xercesc::DOMDocumentType* _typ;

  Node* _head;

  bool _page_started;

  /**
     Set DOMWriter features
  */
  void _setWriterFeatures();

};

#endif

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -

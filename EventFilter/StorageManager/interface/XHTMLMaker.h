// $Id: XHTMLMaker.h,v 1.12 2011/07/07 09:22:44 mommsen Exp $
/// @file: XHTMLMaker.h 

#ifndef EventFilter_StorageManager_XHTMLMaker_h
#define EventFilter_StorageManager_XHTMLMaker_h

#include <xercesc/dom/DOM.hpp>
#include <xercesc/dom/DOMDocument.hpp>
#include <xercesc/dom/DOMWriter.hpp>
#include <xercesc/util/XMLString.hpp>

#include <map>
#include <string>
#include <iostream>
#include <stdint.h>

namespace stor {

  /**
    Helper class to build XHTML pages

    $Author: mommsen $
    $Revision: 1.12 $
    $Date: 2011/07/07 09:22:44 $
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
    Node* getHead() const { return head_; }

    /**
      Add child
    */
    Node* addNode
    (
      const std::string& name,
      Node* parent,
      const AttrMap& attrs
    );

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
      Add an int
    */
    void addInt( Node* parent, const int& );

    /**
      Add an unsigned int
    */
    void addInt( Node* parent, const unsigned int& );

    /**
      Add a long
    */
    void addInt( Node* parent, const long& );

    /**
      Add an unsigned long
    */
    void addInt( Node* parent, const unsigned long& );

    /**
      Add a long long
    */
    void addInt( Node* parent, const long long& );

    /**
      Add an unsigned long long
    */
    void addInt( Node* parent, const unsigned long long& );

    /**
      Add an unsigned long in hex format
    */
    void addHex( Node* parent, const unsigned long& );


    /**
      Add a double
    */
    void addDouble( Node* parent, const double& value, const unsigned int& precision = 2 );

    /**
      Add a boolean
    */
    void addBool( Node* parent, const bool& );

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

    xercesc::DOMDocument* doc_;
    xercesc::DOMWriter* writer_;
    xercesc::DOMDocumentType* typ_;
    
    Node* head_;
    
    bool pageStarted_;

    /**
      Set DOMWriter features
    */
    void setWriterFeatures_();
    
  };

} // namespace stor

#endif // EventFilter_StorageManager_XHTMLMaker_h

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -

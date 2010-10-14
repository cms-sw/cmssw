#ifndef DD_XMLElement_H
#define DD_XMLElement_H
/***************************************************************************
                          DDXMLElement.h  -  description
                             -------------------
    begin                : Fri Mar 15 2002
    email                : case@ucdhep.ucdavis.edu
 ***************************************************************************/

// -------------------------------------------------------------------------
// Includes
// -------------------------------------------------------------------------

#include <map>
#include <string>
#include <vector>

#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Base/interface/DDException.h"

#include "DetectorDescription/Parser/interface/DDLElementRegistry.h"

// -------------------------------------------------------------------------
// Class declaration
// -------------------------------------------------------------------------

/// This is a base class for processing XML elements in the DDD
/** \class DDXMLElement
 *                                                                         
 *           DDXMLElement                       
 *
 *  Component of DDL XML Parsing                                
 *                                                                         
 *  A DDXMLElement stores all the attributes and text of an XML element.
 *  It is designed to accumulate this information unless cleared.  In other
 *  words, it accumulates sets of attributes, and allows the appending of
 *  text indefinitely, as opposed to, say, creating another class which
 *  is designed to hold a std::vector of single element information. This is 
 *  contrary to the way that XML normally defines an element, but for
 *  DDL, this works fine.
 *
 *  One of the things that one needs to build in to each subclass is when
 *  an element needs to be cleared.  For some, emptying the std::vectors should
 *  happen at the end (of the processElement method).  For some, clearing
 *  is ONLY done by the parent.  For example, SpecPar and its child PartSelector.
 *  or Polyhedra and its child ZSection.  In some cases elements can be in one
 *  or more parent elements as well as on their own (Vector, Map).  For these
 *  the processing currently depends on the parent so one must clear only as
 *  appropriate.
 *
 *
 *                                                                         
 */

typedef std::map <std::string, std::string> DDXMLAttribute;
typedef std::map <std::string, std::vector<std::string> > AttrAccumType;

class DDXMLElement
{
  friend std::ostream & operator<<(std::ostream & os, const DDXMLElement & element);

public:

  /// Constructor.
  DDXMLElement( DDLElementRegistry* myreg );

  /// Constructor for autoClear element.
  DDXMLElement( DDLElementRegistry* myreg, const bool& clearme );

  /// Destructor
  virtual ~DDXMLElement( void );
  
  /// Load the element attributes.
  /**
   * The loadAttributes method loads the attributes of the element into a
   * std::map<std::string, std::string> which is used to store Name-Value pairs.  It takes
   * as input two std::vectors of strings containing "synchronized" names and
   * values.
   * 
   * In the SAX2 based calling process, this is done on a startElement event.
   *
   */
  void loadAttributes( const std::string& elemName,
		       const std::vector<std::string> & names,
		       const std::vector<std::string> & values,
		       const std::string& nmspace, DDCompactView& cpv );

  /// Used to load both text and XML comments into this object
  /**
   *
   * At the current time this is done simply as a way for the user of this
   * class to accumulate text and/or comments as std::vector of strings, each one
   * matching the std::vector of attributes.  Therefore loadText starts a new 
   * text storage.
   *
   */
  void loadText( const std::string& inText );

  /// append to the current (i.e. most recently added)
  void appendText( const std::string& inText );

  /// retrieve the text blob.
  const std::string getText( size_t tindex = 0 ) const;

   /// gotText()? kind of like gotMilk?  Yes = text has already been encountered.
  virtual bool gotText( void ) const;

  /// clear this element's contents.
  virtual void clear( void );

  /// Access to attributes by name.
  virtual const std::string & getAttribute( const std::string& name ) const;

  /// Get a "row" of attributes, i.e. one attribute set
  virtual const DDXMLAttribute& getAttributeSet( size_t aIndex = 0 ) const;

  const virtual DDName getDDName( const std::string& defaultNS, const std::string& attname = std::string( "name" ), size_t aIndex = 0 );

/*   /// Gets the value of the name part of an attribute of the form ns:name. */
/*   const virtual std::string getName(const std::string& attname, size_t aIndex = 0); */

/*   /// Gets the namespace of an attribute of the form ns:name. */
/*   virtual std::string getNameSpace(const std::string& defaultNS, const std::string& attname, size_t aIndex = 0); */

  /// Returns a specific value from the aIndex set of attributes.
  virtual const std::string & get( const std::string& name, size_t aIndex = 0 ) const;

  /// Returns a set of values as a std::vector of strings, given the attribute name.
  virtual std::vector<std::string> getVectorAttribute( const std::string& name );

  /// Number of elements accumulated.
  virtual size_t size( void ) const;

  virtual std::vector<DDXMLAttribute>::const_iterator begin( void );

  virtual std::vector<DDXMLAttribute>::const_iterator end( void );
 
  /// Set parent element name to central list of names.
  void setParent( const std::string& pename );

  /// Set self element name to central list of names.
  void setSelf( const std::string& sename );

  /// access to parent element name
  const std::string& parent( void ) const;

  /// Processing the element. 
  /** 
   * The processElement method completes any necessary work to process the XML
   * element.
   *
   * For example, this can be used to call the DDCore to make the geometry in
   * memory.  There is a default for this so that if not declared in the 
   * inheriting class, no processing is done.
   */
  virtual void processElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv );

  /// Called by loadAttributes AFTER attributes are loaded.
  /** 
   * The preProcessElement method can assume that the attributes are loaded and
   * perform any code that is necessary at the start of an element.
   *
   * This would allow users to call their own code to setup anything necessary
   * for the continued processing of the child elements.
   *
   */
  virtual void preProcessElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv );

  /// Allow for the elements to have their own streaming method, but also provide a default.
  virtual void stream( std::ostream & os ) const;

  /// Allow the elements of this type to be iterated over using ++ operator.
  std::vector<DDXMLAttribute>::const_iterator& operator++( int inc );

  /// Have any elements of this type been encountered but not processed?
  virtual bool isEmpty( void ) const;

  /// format std::string for throw an error.
  void throwError( const std::string& keyMessage ) const;

  // protected:
  /// WARNING: abused by other classes in this system: yet another conversion from int to std::string...
  static std::string itostr( int i );

 protected:
  DDLElementRegistry* myRegistry_;

 private:
  /// behind the scenes appending to pAttributes...
  void appendAttributes( std::vector<std::string> & tv, const std::string& name );

  std::vector<DDXMLAttribute> attributes_; // std::vector of name-value std::map (i.e. multiple elements of the same type.
  std::vector<std::string> text_; // accumulates text.. one per element of this type.
  AttrAccumType attributeAccumulator_;  // temporary holder for most recent accessed attributes_... remove later!
  bool autoClear_;
  std::vector<DDXMLAttribute>::const_iterator myIter_;
  std::string myElement_;
  std::string parentElement_;
};

#endif

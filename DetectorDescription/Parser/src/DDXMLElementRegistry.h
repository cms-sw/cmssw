#ifndef DD_XMLElementRegistry_H
#define DD_XMLElementRegistry_H
// -------------------------------------------------------------------------
// Includes
// -------------------------------------------------------------------------
#include "DetectorDescription/Parser/interface/DDXMLElement.h"

#include <map>
#include <string>

// -------------------------------------------------------------------------
// Class declaration
// -------------------------------------------------------------------------

/// This is a base class for Registering DDXMLElements
/** \class DDXMLElementRegistry
 *                                                                         
 *  DDXMLElementRegistry.h  -  description
 *  -------------------
 *  begin: Wed Mar 27 2002
 *  email: case@ucdhep.ucdavis.edu
 *                                                                         
 *  This class is registry of DDXMLElements.  It is used wherever you want 
 *  to register and retrieve DDXMLElements so that using getElement you can
 *  get a pointer to a DDXMLElement based on the name of the XML Element.
 *                                                                         
 */

class DDXMLElementRegistry
{

  friend std::ostream & operator<<(std::ostream & os, const DDXMLElementRegistry & element);

 public:

  typedef std::map <std::string, DDXMLElement*> RegistryMap;

  /// Destructor
  virtual ~DDXMLElementRegistry();
  
  /// This allows other Elements to register themselves with the static registry
  void registerElement(const std::string& name, DDXMLElement*);

  /// THE most important part.  Getting the pointer to a given element type.
  /**
   *  If this is called with a DDXMLElementRegistry pointer, it will simply
   *  return a pointer if already registered or NULL, no instantiating.
   *
   */
  DDXMLElement* getElement(const std::string& name); 

  /// Get the name given a pointer.  This may not be needed...
  std::string getElementName(DDXMLElement* theElement);

  void stream(std::ostream & os) const;

 protected:
 
  /// Private constructor for singleton.
  DDXMLElementRegistry();

/*    /// Way to check getElement returned a correct response. */
/*    RegistryMap::iterator DDXMLElementRegistry::find(const std::string& name) const; */

/*    /// Way to check getElement returned a correct response. */
/*    RegistryMap::iterator DDXMLElementRegistry::end() const; */

 private:
  RegistryMap* registry_;

};

#endif

#ifndef DD_XMLElementRegistry_H
#define DD_XMLElementRegistry_H
// -------------------------------------------------------------------------
// Includes
// -------------------------------------------------------------------------
#include "DDXMLElement.h"

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

/* class DDXMLElementRegistry */
/* { */

/*   friend std::ostream & operator<<(std::ostream & os, const DDXMLElementRegistry & element); */

/*  public: */


/*   /// Destructor */
/*   virtual ~DDXMLElementRegistry(); */
  
/*   void stream(std::ostream & os) const; */

/*  protected: */
 
/*   /// Private constructor for singleton. */
/*   DDXMLElementRegistry(); */

/* /\*    /// Way to check getElement returned a correct response. *\/ */
/* /\*    RegistryMap::iterator DDXMLElementRegistry::find(const std::string& name) const; *\/ */

/* /\*    /// Way to check getElement returned a correct response. *\/ */
/* /\*    RegistryMap::iterator DDXMLElementRegistry::end() const; *\/ */

/*  private: */
/*   RegistryMap registry_; */

/* }; */

#endif

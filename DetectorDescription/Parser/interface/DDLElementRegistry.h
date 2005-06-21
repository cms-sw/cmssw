#ifndef DDL_ElementRegistry_H
#define DDL_ElementRegistry_H
// -------------------------------------------------------------------------
// Includes
// -------------------------------------------------------------------------

#include <string>

#include "DetectorDescription/DDParser/interface/DDXMLElement.h"
#include "DetectorDescription/DDParser/interface/DDXMLElementRegistry.h"

// CLHEP Dependencies
#include "CLHEP/Evaluator/Evaluator.h"

// -------------------------------------------------------------------------
// Class declaration
// -------------------------------------------------------------------------


/// The main class for processing parsed elements.
/** \class DDLElementRegistry
 *                                                                         
 *
 *  DDLElementRegistry.h  -  description
 *  -------------------
 *  begin                : Wed Oct 24 2001
 *  email                : case@ucdhep.ucdavis.edu
 *
 *  This class is designed to serve as a registry of all DDL XML elements.
 *  It inherits from DDXMLElementRegistry.
 *
 *  This class is responsible for constructing and destructing
 *  any necessary DDL element.
 *
 */

class DDLElementRegistry : public DDXMLElementRegistry
{

 public:
  /// Destructor
  virtual ~DDLElementRegistry();
  
  /// This makes it a singleton.
  static DDLElementRegistry* instance();

  static DDXMLElement* getElement(const std::string& name);

 protected:
  /// Private constructor for singleton.
  DDLElementRegistry();

  // private:
  //static DDLElementRegistry* instance_;
  //static std::string defaultElement_;
  //  std::string defaultElement_;
};

#endif

#ifndef DDL_ElementRegistry_H
#define DDL_ElementRegistry_H
// -------------------------------------------------------------------------
// Includes
// -------------------------------------------------------------------------

#include <string>
#include <map>

#include <DetectorDescription/Base/interface/Singleton.h>
#include <DetectorDescription/Base/interface/Singleton.icc>

class DDXMLElement;

// CLHEP Dependencies
#include <CLHEP/Evaluator/Evaluator.h>
#include "DetectorDescription/ExprAlgo/interface/ClhepEvaluator.h"

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

class DDLElementRegistry
{

 public:
  typedef std::map <std::string, DDXMLElement*> RegistryMap;

  /// Constructor
  DDLElementRegistry();

  /// Destructor
  ~DDLElementRegistry();
  
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
  const std::string& getElementName(DDXMLElement* theElement) const;
  ClhepEvaluator &evaluator() { return evaluator_; }

 private:
  RegistryMap     registry_;
  ClhepEvaluator  evaluator_;
};

#endif

#ifndef DDL_ElementRegistry_H
#define DDL_ElementRegistry_H

#include <string>
#include <map>

#include <DetectorDescription/Base/interface/Singleton.h>
#include <DetectorDescription/Base/interface/Singleton.icc>

class DDXMLElement;

#include <CLHEP/Evaluator/Evaluator.h>
#include "DetectorDescription/ExprAlgo/interface/ExprEvalSingleton.h"

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

  DDLElementRegistry();

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
  ClhepEvaluator &evaluator() { return ExprEvalSingleton::instance(); }

 private:
  RegistryMap registry_;
};

///This is only here because of the boost::spirit::parser stuff of DDLMap needing to be re-designed.
typedef DDI::Singleton<DDLElementRegistry> DDLGlobalRegistry;

#endif

#ifndef DETECTOR_DESCRIPTION_PARSER_DDL_ELEMENT_REGISTRY_H
#define DETECTOR_DESCRIPTION_PARSER_DDL_ELEMENT_REGISTRY_H

#include "DetectorDescription/Core/interface/ClhepEvaluator.h"

#include <CLHEP/Evaluator/Evaluator.h>
#include <string>
#include <map>
#include <memory>

class DDXMLElement;

/// The main class for processing parsed elements.
/** \class DDLElementRegistry
 *                                                                         
 *  This class is designed to serve as a registry of all DDL XML elements.
 *
 *  This class is responsible for constructing and destructing
 *  any necessary DDL element.
 *
 */

class DDLElementRegistry
{

 public:
  typedef std::map <std::string, std::shared_ptr<DDXMLElement> > RegistryMap;

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
  std::shared_ptr<DDXMLElement> getElement(const std::string& name); 

  ClhepEvaluator &evaluator() { return evaluator_; }

 private:
  RegistryMap registry_;
  ClhepEvaluator evaluator_;
};

#endif

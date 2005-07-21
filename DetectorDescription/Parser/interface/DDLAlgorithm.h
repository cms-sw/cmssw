#ifndef DDL_ALGORITHM_H
#define DDL_ALGORITHM_H

#include "DetectorDescription/Parser/interface/DDXMLElement.h"

#include <string>

/// DDLAlgorithm processes Algorithm elements.
/** @class DDLAlgorithm
 * @author Michael Case
 *
 *  DDLAlgorithm.h  -  description
 *  -------------------
 *  begin: Saturday November 29, 2003
 *  email: case@ucdhep.ucdavis.edu
 *
 *  This element is used to algorithmically create and position detector
 *  LogicalParts.
 *
 */

class DDLAlgorithm : public DDXMLElement
{
 public:

  /// Constructor
  DDLAlgorithm();

  /// Destructor
  ~DDLAlgorithm();

  void preProcessElement (const std::string& name, const std::string& nmspace); 

  void processElement (const std::string& name, const std::string& nmspace); 

};

#endif


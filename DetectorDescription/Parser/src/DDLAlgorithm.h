#ifndef DDL_ALGORITHM_H
#define DDL_ALGORITHM_H

#include "DDXMLElement.h"

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
  DDLAlgorithm( DDLElementRegistry* myreg );

  /// Destructor
  ~DDLAlgorithm();

  void preProcessElement (const std::string& name, const std::string& nmspace, DDCompactView& cpv); 

  void processElement (const std::string& name, const std::string& nmspace, DDCompactView& cpv); 

};

#endif


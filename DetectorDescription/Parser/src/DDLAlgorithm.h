#ifndef DDL_ALGORITHM_H
#define DDL_ALGORITHM_H

#include <string>

#include "DDXMLElement.h"

class DDCompactView;
class DDLElementRegistry;

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

class DDLAlgorithm final : public DDXMLElement {
public:
  DDLAlgorithm(DDLElementRegistry* myreg);

  void preProcessElement(const std::string& name, const std::string& nmspace, DDCompactView& cpv) override;
  void processElement(const std::string& name, const std::string& nmspace, DDCompactView& cpv) override;
};

#endif

#ifndef DDL_PseudoTrap_H
#define DDL_PseudoTrap_H

#include <string>
#include <vector>

#include "DDLSolid.h"

class DDCompactView;
class DDLElementRegistry;

/** @class DDLPseudoTrap
 * @author Michael Case
 *
 *  DDLPseudotrap.h  -  description
 *  -------------------
 *  begin: Mon Jul 14, 2003
 *  email: case@ucdhep.ucdavis.edu
 *
 * PseudoTrap processor processes PseudoTrap element.
 *
 */
class DDLPseudoTrap final : public DDLSolid {
public:
  DDLPseudoTrap(DDLElementRegistry* myreg);

  void processElement(const std::string& name, const std::string& nmspace, DDCompactView& cpv) override;
};

#endif

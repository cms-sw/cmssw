#ifndef DDLTRAPEZOID_H
#define DDLTRAPEZOID_H

#include <string>
#include <vector>

#include "DDLSolid.h"

class DDCompactView;
class DDLElementRegistry;

/** @class DDLTrapezoid
 * @author Michael Case
 *
 *  DDLTrapezoid.h  -  description
 *  -------------------
 *  begin: Mon Oct 29 2001
 *  email: case@ucdhep.ucdavis.edu
 *
 * Trapezoid processor processes Trapezoid and Trd1 DDL elements.
 *
 */
class DDLTrapezoid final : public DDLSolid {
public:
  DDLTrapezoid(DDLElementRegistry* myreg);

  void processElement(const std::string& name, const std::string& nmspace, DDCompactView& cpv) override;
};
#endif

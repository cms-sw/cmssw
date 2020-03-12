#ifndef DDLTORUS_H
#define DDLTORUS_H

#include <string>
#include <vector>

#include "DDLSolid.h"

class DDCompactView;
class DDLElementRegistry;

/** @class DDLTorus
 * @author Michael Case
 *
 *  DDLTorus.h  -  description
 *  -------------------
 *  begin: Fri May 25 2007
 *  email: case@ucdhep.ucdavis.edu
 *
 * Torus, same as G4Torus
 *
 */
class DDLTorus final : public DDLSolid {
public:
  DDLTorus(DDLElementRegistry* myreg);

  void processElement(const std::string& name, const std::string& nmspace, DDCompactView& cpv) override;
};

#endif

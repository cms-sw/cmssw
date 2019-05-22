#ifndef DDL_Cone_H
#define DDL_Cone_H

#include <string>

// -------------------------------------------------------------------------
// Includes
// -------------------------------------------------------------------------
#include "DDLSolid.h"

class DDCompactView;
class DDLElementRegistry;

/// DDLCone processes all Cone elements.
/** @class DDLCone
 * @author Michael Case
 *                                                  
 *  DDLCone.h  -  description
 *  -------------------
 *  begin: Mon Oct 29 2001
 *  email: case@ucdhep.ucdavis.edu
 *       
 *  This processes DDL Cone and Cons elements.
 *                                                                         
 */

class DDLCone final : public DDLSolid {
public:
  DDLCone(DDLElementRegistry* myreg);

  void processElement(const std::string& name, const std::string& nmspace, DDCompactView& cpv) override;
};

#endif

#ifndef DDL_ElementaryMaterial_H
#define DDL_ElementaryMaterial_H

#include <string>

// -------------------------------------------------------------------------
// Includes
// -------------------------------------------------------------------------
#include "DDLMaterial.h"

class DDCompactView;
class DDLElementRegistry;

/// DDLElementaryMaterial processes ElementaryMaterial elements.
/** @class DDLElementaryMaterial
 * @author Michael Case
 * 
 *  DDLElementaryMaterial.h  -  description
 *  -------------------
 *  begin                : Wed Oct 31 2001
 *  email                : case@ucdhep.ucdavis.edu
 *
 *  A simple or elementary material.  Some systems distinguish between ions
 *  and elements (in the chemical sense).  The DDL and this Parser
 *  deal with them all as ElementaryMaterial elements (in the XML sense).
 *
 */
class DDLElementaryMaterial final : public DDLMaterial {
public:
  DDLElementaryMaterial(DDLElementRegistry* myreg);

  void processElement(const std::string& name, const std::string& nmspace, DDCompactView& cpv) override;
};

#endif

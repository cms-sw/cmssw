#ifndef DDL_CompositeMaterial_H
#define DDL_CompositeMaterial_H

#include <string>

// -------------------------------------------------------------------------
// Includes
// -------------------------------------------------------------------------
#include "DDLMaterial.h"

class DDCompactView;
class DDLElementRegistry;

/// DDLCompositeMaterial processes all CompositeMaterial elements.
/** @class DDLCompositeMaterial.
 * @author Michael Case
 *                                                                         
 *   DDLCompositeMaterial.h  -  description
 *   -------------------
 *   begin: Wed Oct 31 2001
 *   email: case@ucdhep.ucdavis.edu
 * 
 * This is the processor for CompositeMaterial DDL elements.
 *                       
 * The CompositeMaterial is an element that contains other elements.  In 
 * particular, it contains rMaterial elements which are references either  
 * to other Composite or Elementary materials.                             
 *                                                                         
 */

class DDLCompositeMaterial final : public DDLMaterial {
public:
  DDLCompositeMaterial(DDLElementRegistry* myreg);

  void preProcessElement(const std::string& name, const std::string& nmspace, DDCompactView& cpv) override;
  void processElement(const std::string& name, const std::string& nmspace, DDCompactView& cpv) override;
};

#endif

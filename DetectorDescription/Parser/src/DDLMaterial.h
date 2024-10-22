#ifndef DDLMaterial_H
#define DDLMaterial_H

#include <string>

// -------------------------------------------------------------------------
// Includes
// -------------------------------------------------------------------------
#include "DDXMLElement.h"

class DDCompactView;
class DDLElementRegistry;

/// DDLMaterial processes Box elements.
/** @class DDLMaterial
 * @author Michael Case
 *                                                                       
 *  DDLMaterial.h  -  description
 *  -------------------
 *  begin: Fri Oct 04 2002
 *  email: case@ucdhep.ucdavis.edu
 *
 *  This class currently serves one purpose only.  That is to create a
 *  reference to the most recently created Material, no matter whether
 *  it is an ElementaryMaterial or CompositeMaterial.
 *                                                                         
 */

class DDLMaterial : public DDXMLElement {
public:
  DDLMaterial(DDLElementRegistry* myreg);

  virtual void setReference(const std::string& nmspace, DDCompactView& cpv);
};

#endif

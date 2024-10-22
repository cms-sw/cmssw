#ifndef DDLSolid_H
#define DDLSolid_H

#include <string>

// -------------------------------------------------------------------------
// Includes
// -------------------------------------------------------------------------
#include "DDXMLElement.h"

class DDCompactView;
class DDLElementRegistry;

/// DDLSolid processes Box elements.
/** @class DDLSolid
 * @author Michael Case
 *                                                                       
 *  DDLSolid.h  -  description
 *  -------------------
 *  begin: Thu Oct 03, 2002
 *  email: case@ucdhep.ucdavis.edu
 *
 *  This class currently serves one purpose only.  That is to create a
 *  reference to the most recently processed DDLSolid, no matter whether
 *  it is an Box, Boolean, Cone, Cons, Polyhedra, Polycone, Reflection,
 *  Shapeless, Trapezoid, Trd1, Tube or Tubs!
 *                                                                         
 */

class DDLSolid : public DDXMLElement {
public:
  /// Constructor
  DDLSolid(DDLElementRegistry* myreg);

  void setReference(const std::string& nmspace, DDCompactView& cpv);
};
#endif

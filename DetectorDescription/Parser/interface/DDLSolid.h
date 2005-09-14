#ifndef DDLSolid_H
#define DDLSolid_H

// -------------------------------------------------------------------------
// Includes
// -------------------------------------------------------------------------
#include "DetectorDescription/Parser/interface/DDXMLElement.h"

#include <string>

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

class DDLSolid : public DDXMLElement
{
 public:

  /// Constructor
  DDLSolid();

  /// Destructor
  ~DDLSolid();

  void setReference (const std::string& nmspace);

};
#endif

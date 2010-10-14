#ifndef DDLTRAPEZOID_H
#define DDLTRAPEZOID_H

// -------------------------------------------------------------------------
// Includes
// -------------------------------------------------------------------------
#include "DDLSolid.h"

#include <string>
#include <vector>

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
class DDLTrapezoid : public DDLSolid
{
public:

  /// Constructor
  DDLTrapezoid( DDLElementRegistry* myreg );

  /// Destructor
  ~DDLTrapezoid( void );

  void processElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv );
};
#endif

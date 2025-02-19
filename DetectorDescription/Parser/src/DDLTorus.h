#ifndef DDLTORUS_H
#define DDLTORUS_H

// -------------------------------------------------------------------------
// Includes
// -------------------------------------------------------------------------
#include "DDLSolid.h"

#include <string>
#include <vector>

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
class DDLTorus : public DDLSolid
{
public:

  /// Constructor
  DDLTorus( DDLElementRegistry* myreg );

  /// Destructor
  ~DDLTorus( void );

  void processElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv ); 
};
#endif

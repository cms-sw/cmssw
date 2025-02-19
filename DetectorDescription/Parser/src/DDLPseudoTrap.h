#ifndef DDL_PseudoTrap_H
#define DDL_PseudoTrap_H

// -------------------------------------------------------------------------
// Includes
// -------------------------------------------------------------------------
#include "DDLSolid.h"

#include <string>
#include <vector>

/** @class DDLPseudoTrap
 * @author Michael Case
 *
 *  DDLPseudotrap.h  -  description
 *  -------------------
 *  begin: Mon Jul 14, 2003
 *  email: case@ucdhep.ucdavis.edu
 *
 * PseudoTrap processor processes PseudoTrap element.
 *
 */
class DDLPseudoTrap : public DDLSolid
{
public:

  /// Constructor
  DDLPseudoTrap( DDLElementRegistry* myreg );

  /// Destructor
  ~DDLPseudoTrap( void );

  void processElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv ); 
};
#endif

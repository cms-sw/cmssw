#ifndef DDL_PseudoTrap_H
#define DDL_PseudoTrap_H

#include <string>
#include <vector>

// -------------------------------------------------------------------------
// Includes
// -------------------------------------------------------------------------
#include "DDLSolid.h"

class DDCompactView;
class DDLElementRegistry;

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

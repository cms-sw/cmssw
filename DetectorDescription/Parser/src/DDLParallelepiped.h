#ifndef DDL_Parallelepiped_H
#define DDL_Parallelepiped_H

// -------------------------------------------------------------------------
// Includes
// -------------------------------------------------------------------------
#include "DDLSolid.h"

#include <string>

/// DDLParallelepiped processes all Parallelepiped elements.
/** @class DDLParallelepiped
 * @author Michael Case
 *                                                  
 *  DDLParallelepiped.h  -  description
 *  -------------------
 *  begin: Mon Oct 29 2001
 *  email: case@ucdhep.ucdavis.edu
 *       
 *  This processes DDL Parallelepiped elements.
 *                                                                         
 */

class DDLParallelepiped : public DDLSolid
{
public:

  /// Constructor
  DDLParallelepiped( DDLElementRegistry* myreg );

  /// Destructor
  ~DDLParallelepiped( void );

  void processElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv ); 
};
#endif

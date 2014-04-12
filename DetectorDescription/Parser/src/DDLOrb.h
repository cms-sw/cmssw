#ifndef DDL_Orb_H
#define DDL_Orb_H

// -------------------------------------------------------------------------
// Includes
// -------------------------------------------------------------------------
#include "DDLSolid.h"

#include <string>

/// DDLOrb processes all Orb elements.
/** @class DDLOrb
 * @author Michael Case
 *                                                  
 *  DDLOrb.h  -  description
 *  -------------------
 *  begin: Mon Oct 29 2001
 *  email: case@ucdhep.ucdavis.edu
 *       
 *  This processes DDL Orb elements.
 *                                                                         
 */

class DDLOrb : public DDLSolid
{
public:

  /// Constructor
  DDLOrb( DDLElementRegistry* myreg );

  /// Destructor
  ~DDLOrb( void );

  void processElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv ); 
};
#endif

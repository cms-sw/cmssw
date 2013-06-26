#ifndef DDL_Ellipsoid_H
#define DDL_Ellipsoid_H

// -------------------------------------------------------------------------
// Includes
// -------------------------------------------------------------------------
#include "DDLSolid.h"

#include <string>

/// DDLEllipsoid processes all Ellipsoid elements.
/** @class DDLEllipsoid
 * @author Michael Case
 *                                                  
 *  DDLEllipsoid.h  -  description
 *  -------------------
 *  begin: Mon Oct 29 2001
 *  email: case@ucdhep.ucdavis.edu
 *       
 *  This processes DDL Ellipsoid elements.
 *                                                                         
 */

class DDLEllipsoid : public DDLSolid
{
public:

  /// Constructor
  DDLEllipsoid( DDLElementRegistry* myreg );

  /// Destructor
  ~DDLEllipsoid( void );

  void processElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv ); 
};
#endif

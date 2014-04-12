#ifndef DDL_Sphere_H
#define DDL_Sphere_H

// -------------------------------------------------------------------------
// Includes
// -------------------------------------------------------------------------
#include "DDLSolid.h"

#include <string>

/// DDLSphere processes all Sphere elements.
/** @class DDLSphere
 * @author Michael Case
 *                                                  
 *  DDLSphere.h  -  description
 *  -------------------
 *  begin: Mon Oct 29 2001
 *  email: case@ucdhep.ucdavis.edu
 *       
 *  This processes DDL Sphere and Cons elements.
 *                                                                         
 */

class DDLSphere : public DDLSolid
{
public:

  /// Constructor
  DDLSphere( DDLElementRegistry* myreg );

  /// Destructor
  ~DDLSphere( void );

  void processElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv ); 
};
#endif

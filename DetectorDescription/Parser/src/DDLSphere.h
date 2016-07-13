#ifndef DDL_Sphere_H
#define DDL_Sphere_H

#include <string>

// -------------------------------------------------------------------------
// Includes
// -------------------------------------------------------------------------
#include "DDLSolid.h"

class DDCompactView;
class DDLElementRegistry;

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

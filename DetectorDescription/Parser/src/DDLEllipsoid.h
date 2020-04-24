#ifndef DDL_Ellipsoid_H
#define DDL_Ellipsoid_H

#include <string>

#include "DDLSolid.h"

class DDCompactView;
class DDLElementRegistry;

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

class DDLEllipsoid final : public DDLSolid
{
public:

  DDLEllipsoid( DDLElementRegistry* myreg );

  void processElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv ) override; 
};

#endif

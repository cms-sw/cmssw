#ifndef DDL_Parallelepiped_H
#define DDL_Parallelepiped_H

#include <string>
#include "DDLSolid.h"

class DDCompactView;
class DDLElementRegistry;

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

class DDLParallelepiped final : public DDLSolid
{
 public:

  DDLParallelepiped( DDLElementRegistry* myreg );

  void processElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv ) override; 
};

#endif

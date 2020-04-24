#ifndef DDL_Orb_H
#define DDL_Orb_H

#include <string>
#include "DDLSolid.h"

class DDCompactView;
class DDLElementRegistry;

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

class DDLOrb final : public DDLSolid
{
 public:

  DDLOrb( DDLElementRegistry* myreg );

  void processElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv ) override; 
};

#endif

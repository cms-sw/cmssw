#ifndef DDL_EllipticalTube_H
#define DDL_EllipticalTube_H

// -------------------------------------------------------------------------
// Includes
// -------------------------------------------------------------------------
#include "DDLSolid.h"

#include <string>

/// DDLEllipticalTube processes all EllipticalTube elements.
/** @class DDLEllipticalTube
 * @author Michael Case
 *                                                  
 *  DDLEllipticalTube.h  -  description
 *  -------------------
 *  begin: Mon Oct 29 2001
 *  email: case@ucdhep.ucdavis.edu
 *       
 *  This processes DDL EllipticalTube elements.
 *                                                                         
 */

class DDLEllipticalTube : public DDLSolid
{
public:

  /// Constructor
  DDLEllipticalTube( DDLElementRegistry* myreg );

  /// Destructor
  ~DDLEllipticalTube( void );

  void processElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv ); 
};
#endif

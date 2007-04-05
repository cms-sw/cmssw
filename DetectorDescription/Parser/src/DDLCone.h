#ifndef DDL_Cone_H
#define DDL_Cone_H

// -------------------------------------------------------------------------
// Includes
// -------------------------------------------------------------------------
#include "DetectorDescription/Parser/interface/DDLSolid.h"

#include <string>

/// DDLCone processes all Cone elements.
/** @class DDLCone
 * @author Michael Case
 *                                                  
 *  DDLCone.h  -  description
 *  -------------------
 *  begin: Mon Oct 29 2001
 *  email: case@ucdhep.ucdavis.edu
 *       
 *  This processes DDL Cone and Cons elements.
 *                                                                         
 */

class DDLCone : public DDLSolid
{
 public:

  /// Constructor
  DDLCone();

  /// Destructor
  ~DDLCone();

  void processElement (const std::string& name, const std::string& nmspace); 

};
#endif

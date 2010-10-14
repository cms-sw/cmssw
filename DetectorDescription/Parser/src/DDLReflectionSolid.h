#ifndef DDL_ReflectionSolid_H
#define DDL_ReflectionSolid_H
// -------------------------------------------------------------------------
// Includes
// -------------------------------------------------------------------------
#include "DDLSolid.h"

#include <string>

/// DDLReflectionSolid processes ReflectionSolid elements.
/** @class DDLReflectionSolid
 * @author Michael Case
 *                                                                       
 *  DDLReflectionSolid.h  -  description
 *  -------------------
 *  begin: Mon Mar 4, 2002
 *  email: case@ucdhep.ucdavis.edu
 *                                                                         
 *  This is the ReflectionSolid processor.
 *                                                                         
 */

class DDLReflectionSolid : public DDLSolid
{
public:

  /// Constructor
  DDLReflectionSolid( DDLElementRegistry* myreg );

  /// Destructor
  ~DDLReflectionSolid( void );

  void preProcessElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv );
  void processElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv );
};
#endif

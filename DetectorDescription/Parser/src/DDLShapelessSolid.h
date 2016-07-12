#ifndef DDL_ShapelessSolid_H
#define DDL_ShapelessSolid_H

#include <string>

// -------------------------------------------------------------------------
// Includes
// -------------------------------------------------------------------------
#include "DDLSolid.h"

class DDCompactView;
class DDLElementRegistry;

/// DDLShapelessSolid processes ShapelessSolid elements.
/** @class DDLShapelessSolid
 * @author Michael Case
 *                                                                       
 *  DDLShapelessSolid.h  -  description
 *  -------------------
 *  begin: Wed May 15 2002
 *  email: case@ucdhep.ucdavis.edu
 *
 *  This is the ShapelessSolid processor.
 *                                                                         
 */

class DDLShapelessSolid : public DDLSolid
{
public:

  /// Constructor
  DDLShapelessSolid( DDLElementRegistry* myreg );

  /// Destructor
  ~DDLShapelessSolid( void );

  void processElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv );

  void preProcessElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv );
};
#endif

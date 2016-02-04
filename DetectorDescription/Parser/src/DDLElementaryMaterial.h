#ifndef DDL_ElementaryMaterial_H
#define DDL_ElementaryMaterial_H

// -------------------------------------------------------------------------
// Includes
// -------------------------------------------------------------------------
#include "DDLMaterial.h"

#include <string>

/// DDLElementaryMaterial processes ElementaryMaterial elements.
/** @class DDLElementaryMaterial
 * @author Michael Case
 * 
 *  DDLElementaryMaterial.h  -  description
 *  -------------------
 *  begin                : Wed Oct 31 2001
 *  email                : case@ucdhep.ucdavis.edu
 *
 *  A simple or elementary material.  Some systems distinguish between ions
 *  and elements (in the chemical sense).  The DDL and this Parser
 *  deal with them all as ElementaryMaterial elements (in the XML sense).
 *
 */
class DDLElementaryMaterial : public DDLMaterial
{
public:

  /// Constructor
  DDLElementaryMaterial( DDLElementRegistry* myreg );

  /// Destructor
  ~DDLElementaryMaterial( void );

  void processElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv ); 
};
#endif

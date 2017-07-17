#ifndef DDL_ReflectionSolid_H
#define DDL_ReflectionSolid_H

#include <string>
#include "DDLSolid.h"

class DDCompactView;
class DDLElementRegistry;

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

class DDLReflectionSolid final : public DDLSolid
{
 public:

  DDLReflectionSolid( DDLElementRegistry* myreg );

  void preProcessElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv ) override;
  void processElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv ) override;
};

#endif

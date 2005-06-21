#ifndef DDL_ReflectionSolid_H
#define DDL_ReflectionSolid_H
// -------------------------------------------------------------------------
// Includes
// -------------------------------------------------------------------------
#include "DetectorDescription/DDParser/interface/DDLSolid.h"

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
  DDLReflectionSolid();

  /// Destructor
  ~DDLReflectionSolid();

  void preProcessElement (const std::string& name, const std::string& nmspace);
  void processElement (const std::string& name, const std::string& nmspace);

};
#endif

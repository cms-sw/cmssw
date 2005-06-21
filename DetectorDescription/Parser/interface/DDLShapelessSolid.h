#ifndef DDL_ShapelessSolid_H
#define DDL_ShapelessSolid_H

// -------------------------------------------------------------------------
// Includes
// -------------------------------------------------------------------------
#include "DetectorDescription/DDParser/interface/DDLSolid.h"

#include <string>

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
  DDLShapelessSolid();

  /// Destructor
  ~DDLShapelessSolid();

  void processElement (const std::string& name, const std::string& nmspace);

  void preProcessElement (const std::string& name, const std::string& nmspace);
};
#endif

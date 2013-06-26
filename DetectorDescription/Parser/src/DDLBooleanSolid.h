#ifndef DDL_BooleanSolid_H
#define DDL_BooleanSolid_H

#include "DDLSolid.h"

#include <string>

/// This class takes care of processing all BooleanSolid type elements.
/** @class DDLBooleanSolid
 * @author Michael Case
 *                                                                         
 *  DDLBooleanSolid.h  -  description
 *  -------------------
 *  begin: Wed Dec 12, 2001
 *  email: case@ucdhep.ucdavis.edu
 *                                                                         
 *  This is the Intersection, Subtraction and Union processor.
 *  A BooleanSolid handles all of these because as far as the DDL is
 *  concerned, they have the same basic form, including two solid 
 *  references, and potentially one translation and one rotation.
 *                                                                         
**/

class DDLBooleanSolid : public DDLSolid
{
 public:

  /// Constructor
  DDLBooleanSolid( DDLElementRegistry* myreg );

  /// Destructor
  ~DDLBooleanSolid();

  void preProcessElement (const std::string& name, const std::string& nmspace, DDCompactView& cpv);

  void processElement (const std::string& name, const std::string& nmspace, DDCompactView& cpv); 

 private:
  std::string dumpBooleanSolid (const std::string& name, const std::string& nmspace); 

};

#endif

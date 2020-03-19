#ifndef DDL_BooleanSolid_H
#define DDL_BooleanSolid_H

#include <string>

#include "DDLSolid.h"

class DDCompactView;
class DDLElementRegistry;

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

class DDLBooleanSolid final : public DDLSolid {
public:
  DDLBooleanSolid(DDLElementRegistry* myreg);

  void preProcessElement(const std::string& name, const std::string& nmspace, DDCompactView& cpv) override;
  void processElement(const std::string& name, const std::string& nmspace, DDCompactView& cpv) override;

private:
  std::string dumpBooleanSolid(const std::string& name, const std::string& nmspace);
};

#endif

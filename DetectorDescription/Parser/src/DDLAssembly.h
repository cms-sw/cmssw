#ifndef DDL_Assembly_H
#define DDL_Assembly_H

#include <string>

#include "DDLSolid.h"

class DDCompactView;
class DDLElementRegistry;

/// DDLAssembly processes Assembly elements.
/** @class DDLAssembly
 * @author Ianna Osborne
 *                                                                       
 *  DDLAssembly.h  -  description
 *
 *  This is the Assembly processor.
 *                                                                         
 */

class DDLAssembly final : public DDLSolid {
public:
  DDLAssembly(DDLElementRegistry* myreg);

  void processElement(const std::string& name, const std::string& nmspace, DDCompactView& cpv) override;
  void preProcessElement(const std::string& name, const std::string& nmspace, DDCompactView& cpv) override;
};

#endif

#ifndef DDLBox_H
#define DDLBox_H

#include <string>

// -------------------------------------------------------------------------
// Includes
// -------------------------------------------------------------------------
#include "DDLSolid.h"

class DDCompactView;
class DDLElementRegistry;

/// DDLBox processes Box elements.
/** @class DDLBox
 * @author Michael Case
 *                                                                       
 *  DDLBox.h  -  description
 *  -------------------
 *  begin: Wed Oct 24 2001
 *  email: case@ucdhep.ucdavis.edu
 *                                                                         
 * This is the Box element processor.
 *                                                                         
 */

class DDLBox final : public DDLSolid {
public:
  DDLBox(DDLElementRegistry* myreg);

  void processElement(const std::string& name, const std::string& nmspace, DDCompactView& cpv) override;
};

#endif

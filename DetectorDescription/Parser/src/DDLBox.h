#ifndef DDLBox_H
#define DDLBox_H

// -------------------------------------------------------------------------
// Includes
// -------------------------------------------------------------------------
#include "DDLSolid.h"

#include <string>

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

class DDLBox : public DDLSolid
{
 public:

  /// Constructor
  DDLBox( DDLElementRegistry* myreg );

  /// Destructor
  ~DDLBox();

  void processElement (const std::string& name, const std::string& nmspace, DDCompactView& cpv);

};
#endif

#ifndef DDLTUBS_H
#define DDLTUBS_H

#include "DDLSolid.h"

#include <string>
#include <vector>

/// DDLTubs processes Tubs elements.
/** @class DDLTubs
 * @author Michael Case
 *
 *  DDLTubs.h  -  description
 *  -------------------
 *  begin: Mon Oct 29 2001
 *  email: case@ucdhep.ucdavis.edu
 *
 *  Tube and Tubs elements are handled by this processor.
 *
 */

class DDLTubs : public DDLSolid
{
public:

  /// Constructor
  DDLTubs( DDLElementRegistry* myreg );

  /// Destructor
  ~DDLTubs( void );

  void processElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv ); 
};

#endif

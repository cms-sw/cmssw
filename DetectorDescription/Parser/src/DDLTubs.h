#ifndef DDLTUBS_H
#define DDLTUBS_H

#include <string>
#include <vector>

#include "DDLSolid.h"

class DDCompactView;
class DDLElementRegistry;

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

class DDLTubs final : public DDLSolid
{
 public:

  DDLTubs( DDLElementRegistry* myreg );

  void processElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv ) override; 
};

#endif

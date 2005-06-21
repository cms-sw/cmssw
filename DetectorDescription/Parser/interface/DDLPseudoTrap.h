#ifndef DDL_PseudoTrap_H
#define DDL_PseudoTrap_H

// -------------------------------------------------------------------------
// Includes
// -------------------------------------------------------------------------
#include "DetectorDescription/DDParser/interface/DDLSolid.h"

#include <string>
#include <vector>

namespace std{} using namespace std;

/** @class DDLPseudoTrap
 * @author Michael Case
 *
 *  DDLPseudotrap.h  -  description
 *  -------------------
 *  begin: Mon Jul 14, 2003
 *  email: case@ucdhep.ucdavis.edu
 *
 * PseudoTrap processor processes PseudoTrap element.
 *
 */
class DDLPseudoTrap : public DDLSolid
{
 public:

  /// Constructor
  DDLPseudoTrap();

  /// Destructor
  ~DDLPseudoTrap();

  void processElement (const std::string& name, const std::string& nmspace); 

};
#endif

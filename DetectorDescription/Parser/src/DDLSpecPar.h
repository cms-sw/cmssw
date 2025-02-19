#ifndef DDLSPECPAR_H
#define DDLSPECPAR_H

#include "DDXMLElement.h"

#include <string>

/// DDLSpecPar processes SpecPar elements.
/** @class DDLSpecPar
 * @author Michael Case
 *
 *  DDLSpecPar.h  -  description
 *  -------------------
 *  begin: Tue Nov 21 2001
 *  email: case@ucdhep.ucdavis.edu
 *
 *  This element is used to specify parameters for a part in the detector.
 *  PartSelector provides a way to associate Parameters with specific parts
 *  of the detector.
 *
 */

class DDLSpecPar : public DDXMLElement
{
public:

  /// Constructor
  DDLSpecPar( DDLElementRegistry* myreg );

  /// Destructor
  ~DDLSpecPar( void );

  void processElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv ); 
};

#endif


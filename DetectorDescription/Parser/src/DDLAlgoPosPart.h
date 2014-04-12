#ifndef DDL_AlgoPosPart_H
#define DDL_AlgoPosPart_H
/***************************************************************************
 ***************************************************************************/
#include "DDXMLElement.h"

#include <string>

/// DDLAlgoPosPart handles AlgoPosPart elements.
/** @class DDLAlgoPosPart
 * @author Michael Case
 *
 *   DDLAlgoPosPart.h  -  description
 *   -------------------
 *   begin: Wed Apr 17 2002
 *   email: case@ucdhep.ucdavis.edu
 *
 *  An AlgoPosPart (or Algorithmic Positioning Part) is used to repeatedly 
 *  position a LogicalPart somewhere inside it's parent.  So, an AlgoPosPart
 *  needs two rLogicalParts (self and parent) on which to operate, an 
 *  Algorithm and it's parameters.
 *
 */

class DDLAlgoPosPart : public DDXMLElement
{
 public:

  /// Constructor
  DDLAlgoPosPart( DDLElementRegistry* myreg );

  /// Destructor
  ~DDLAlgoPosPart();

  void processElement (const std::string& name, const std::string& nmspace, DDCompactView& cpv); 
};

#endif

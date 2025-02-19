#ifndef DDL_PosPart_H
#define DDL_PosPart_H

#include "DDXMLElement.h"

#include <string>

/// DDLPosPart handles PosPart elements.
/** @class DDLPosPart
 * @author Michael Case
 *
 *  DDLPosPart.h  -  description
 *  -------------------
 *  begin: Tue Oct 30 2001
 *  email: case@ucdhep.ucdavis.edu
 *
 *  A PosPart (or Positioning Part or Part Positioner :-)) is used to
 *  position a LogicalPart somewhere inside it's parent.  So, A PosPart
 *  needs two rLogicalParts (self and parent) on which to operate, a
 *  Translation and a Rotation. 
 *
 */

class DDLPosPart : public DDXMLElement
{
public:

  /// Constructor
  DDLPosPart( DDLElementRegistry* myreg );

  /// Destructor
  ~DDLPosPart( void );

  void preProcessElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv ); 

  void processElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv ); 
};

#endif

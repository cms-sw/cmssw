#ifndef DDL_RotationSequence_H
#define DDL_RotationSequence_H

// -------------------------------------------------------------------------
// Includes
// -------------------------------------------------------------------------
#include "DDLRotationByAxis.h"

#include <string>

///  DDLRotationSequence handles a set of Rotations.
/** @class DDLRotationSequence
 * @author Michael Case
 *
 *  DDLRotationSequence.h  -  description
 *  -------------------
 *  begin: Friday Nov. 15, 2003
 *  email: case@ucdhep.ucdavis.edu
 *
 *
 *  This is the RotationSequence processor.
 *
 */
class DDLRotationSequence : public DDLRotationByAxis
{
public:

  /// Constructor 
  DDLRotationSequence( DDLElementRegistry* myreg );

  /// Destructor
  ~DDLRotationSequence( void );

  void preProcessElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv );

  void processElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv );
};

#endif

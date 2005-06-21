#ifndef DDL_RotationSequence_H
#define DDL_RotationSequence_H

// -------------------------------------------------------------------------
// Includes
// -------------------------------------------------------------------------
#include "DetectorDescription/DDParser/interface/DDLRotationByAxis.h"

#include <string>

//namespace ddl {

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
  DDLRotationSequence();

  /// Destructor
  ~DDLRotationSequence();

  void preProcessElement (const std::string& name, const std::string& nmspace);

  void processElement (const std::string& name, const std::string& nmspace);

 private:

};

//} 
#endif

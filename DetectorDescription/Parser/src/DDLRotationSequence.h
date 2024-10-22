#ifndef DDL_RotationSequence_H
#define DDL_RotationSequence_H

#include <string>

#include "DDLRotationByAxis.h"

class DDCompactView;
class DDLElementRegistry;

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
class DDLRotationSequence final : public DDLRotationByAxis {
public:
  DDLRotationSequence(DDLElementRegistry* myreg);

  void preProcessElement(const std::string& name, const std::string& nmspace, DDCompactView& cpv) override;
  void processElement(const std::string& name, const std::string& nmspace, DDCompactView& cpv) override;
};

#endif

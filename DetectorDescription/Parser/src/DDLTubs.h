#ifndef DETECTOR_DESCRIPTION_PARSER_DDLTUBS_H
#define DETECTOR_DESCRIPTION_PARSER_DDLTUBS_H

#include <string>
#include <vector>

#include "DDLSolid.h"

class DDCompactView;
class DDLElementRegistry;

/// DDLTubs processes Tubs elements.
/** @class DDLTubs
 *
 *  DDLTubs.h  -  description
 *
 *  Tube and Tubs elements are handled by this processor.
 *
 */

class DDLTubs final : public DDLSolid {
public:
  DDLTubs(DDLElementRegistry* myreg);

  void processElement(const std::string& name, const std::string& nmspace, DDCompactView& cpv) override;
};

#endif

#ifndef DETECTOR_DESCRIPTION_PARSER_DDL_PGON_GENERATOR_H
#define DETECTOR_DESCRIPTION_PARSER_DDL_PGON_GENERATOR_H

#include <string>

#include "DDLSolid.h"

class DDCompactView;
class DDLElementRegistry;

/// DDLPgonGenerator processes DDL XML Extruded Polygone elements.
/** 
 *  The PgonGenerator element uses XYPoint elements. The DDLXYPoint
 *  can return the x and y std::vectors with the points needed to form the
 *  extruded polygone.
 *
 */

class DDLPgonGenerator final : public DDLSolid {
public:
  DDLPgonGenerator(DDLElementRegistry* myreg);

  void processElement(const std::string& name, const std::string& nmspace, DDCompactView& cpv) override;
  void preProcessElement(const std::string& name, const std::string& nmspace, DDCompactView& cpv) override;
};

#endif

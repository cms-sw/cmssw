#ifndef DETECTOR_DESCRIPTION_PARSER_DDL_MULTI_UNION_SOLID_H
#define DETECTOR_DESCRIPTION_PARSER_DDL_MULTI_UNION_SOLID_H

#include <string>

#include "DDLSolid.h"

class DDCompactView;
class DDLElementRegistry;

/// This class takes care of processing all MultiUnionSolid type elements.
/** @class DDLMultiUnionSolid
 *                                                                         
 *  This is the MultiUnion processor.
 *                                                                         
**/

class DDLMultiUnionSolid final : public DDLSolid
{
 public:

  DDLMultiUnionSolid( DDLElementRegistry* myreg );

  void preProcessElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv ) override;
  void processElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv ) override; 

 private:
  std::string dumpMultiUnionSolid (const std::string& name, const std::string& nmspace); 
};

#endif

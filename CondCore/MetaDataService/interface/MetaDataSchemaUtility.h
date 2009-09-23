#ifndef CondCore_MetaDataSchemaUtility_h
#define CondCore_MetaDataSchemaUtility_h

#include "CondCore/DBCommon/interface/DbSession.h"

namespace cond{
  class MetaDataSchemaUtility{
  public:
    MetaDataSchemaUtility(cond::DbSession& coraldb);
    ~MetaDataSchemaUtility();
    /// create metadata tables if not existing
    void create();
    /// drop metadata tables is existing
    void drop();
  private:
    cond::DbSession m_coraldb;
  };
}//ns cond
#endif

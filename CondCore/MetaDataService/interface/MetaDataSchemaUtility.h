#ifndef CondCore_MetaDataSchemaUtility_h
#define CondCore_MetaDataSchemaUtility_h

namespace cond{
  class CoralTransaction;
  class MetaDataSchemaUtility{
  public:
    MetaDataSchemaUtility(CoralTransaction& coraldb);
    ~MetaDataSchemaUtility();
    /// create metadata tables if not existing
    void create();
    /// drop metadata tables is existing
    void drop();
  private:
    cond::CoralTransaction& m_coraldb;
  };
}//ns cond
#endif

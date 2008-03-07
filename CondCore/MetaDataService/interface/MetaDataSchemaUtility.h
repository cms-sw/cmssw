#ifndef CondCore_MetaDataSchemaUtility_h
#define CondCore_MetaDataSchemaUtility_h

namespace cond{
  class CoralTransaction;
  class MetaDataSchemaUtility{
  public:
    MetaDataSchemaUtility(const CoralTransaction& coraldb);
    /// create metadata tables if not existing
    void create();
    /// drop metadata tables is existing
    void drop();
  private:
    
  };
}//ns cond
#endif

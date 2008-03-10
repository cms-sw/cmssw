#ifndef CondCore_IOVSchemaUtility_h
#define CondCore_IOVSchemaUtility_h

namespace cond{
  class CoralTransaction;
  class IOVSchemaUtility{
  public:
    IOVSchemaUtility(CoralTransaction& coraldb);
    ~IOVSchemaUtility();
    /// create iov tables if not existing
    void create();
    /// drop iov tables if existing
    void drop();
    /// truncate iov tables if existing
    void truncate();
  private:
    cond::CoralTransaction& m_coraldb;
  };
}//ns cond
#endif

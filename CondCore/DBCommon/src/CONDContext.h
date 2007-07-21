#ifndef DBCommon_CONDContext_h
#define DBCommon_CONDContext_h
namespace seal{
  class Context; 
}
namespace cond{
  class CONDContext{
  public:
    static seal::Context* getPOOLContext();
    static seal::Context* getOwnContext();
  };
}// ns cond
#endif

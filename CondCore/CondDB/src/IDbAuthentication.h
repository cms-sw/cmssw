#ifndef CondCore_CondDB_IDbAuthentication_h
#define CondCore_CondDB_IDbAuthentication_h

#include <string>

namespace cond {

  namespace persistency {

    class IDbAuthentication {
    public:
      virtual ~IDbAuthentication() {}
      virtual std::string principalName() = 0;
    };
  }  // namespace persistency
}  // namespace cond
#endif

#ifndef CondCore_CondDB_CondServiceWrapper_h
#define CondCore_CondDB_CondServiceWrapper_h

#include <string>

namespace coral {
  class Service;
}

/**
 * The wrapper is used to allow edm::PluginFactory to change its
 * return type to unique_ptr from a raw pointer. The unique_ptr does
 * not work for coral::Service, because its destructor is protected
 * and ownership is managed by intrusive reference counting.
 */
namespace cond {
  struct CoralServiceWrapperBase {
    virtual ~CoralServiceWrapperBase() = default;
    virtual coral::Service* create(const std::string& componentname) const = 0;
  };

  template <typename T>
  struct CoralServiceWrapper : public CoralServiceWrapperBase {
    ~CoralServiceWrapper() override = default;
    coral::Service* create(const std::string& componentname) const override { return new T{componentname}; }
  };
}  // namespace cond

#endif

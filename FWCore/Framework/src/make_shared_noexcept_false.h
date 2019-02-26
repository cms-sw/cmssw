#ifndef FWCore_Framework_make_shared_noexcept_false_h
#define FWCore_Framework_make_shared_noexcept_false_h
#include <memory>
namespace edm {
template<typename T, typename ... Args>
std::shared_ptr<T> make_shared_noexcept_false(Args&& ... args) {
#if defined(__APPLE__)
// libc++ from Apple Clang does not allow non-default destructors 
// in some cases the destructor uses noexcept(false).
return std::shared_ptr<T>( new T(std::forward<Args>(args)... ));
#else
return std::make_shared<T>(std::forward<Args>(args)... );
#endif
} 
}
#endif

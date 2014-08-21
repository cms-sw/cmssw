#ifndef DDI_Singleton_h
#define DDI_Singleton_h

#include <memory>
#include <mutex>

namespace DDI {
 template <class I> 
 class Singleton 
 {
 public:
   typedef I value_type;
   virtual ~Singleton() {}
   static value_type & instance();
   
 private:
   static std::unique_ptr<value_type> m_instance;
   static std::once_flag m_onceFlag;
   Singleton(void);
   Singleton(const Singleton&);
   Singleton& operator=(const Singleton &);
 };

 template<class I> std::unique_ptr<I> Singleton<I>::m_instance;
 template<class I> std::once_flag Singleton<I>::m_onceFlag;
}
#endif

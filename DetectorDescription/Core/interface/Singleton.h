#ifndef DDI_Singleton_h
#define DDI_Singleton_h

namespace DDI {
 template <class I> 
 class Singleton 
 {
 public:
   typedef I value_type;
   virtual ~Singleton() {}
   static value_type & instance();
   
 private:
   Singleton(void) = delete;
   Singleton(const Singleton&) = delete;
   Singleton& operator=(const Singleton &) = delete;
 };
}
#endif

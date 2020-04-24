#ifndef DETECTOR_DESCRIPTION_CORE_DDI_SINGLETON_H
#define DETECTOR_DESCRIPTION_CORE_DDI_SINGLETON_H

namespace DDI {
 template <class I> 
 class Singleton 
 {
 public:
   typedef I value_type;
   virtual ~Singleton() {}
   static value_type & instance();
   
   Singleton(void) = delete;
   Singleton(const Singleton&) = delete;
   Singleton& operator=(const Singleton &) = delete;
 };
}

#endif

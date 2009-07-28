#ifndef DDI_Singleton_h
#define DDI_Singleton_h
#include "DetectorDescription/Base/interface/Ptr.h"

namespace DDI {
 template <class I> 
 class Singleton //: private I
 {
 public:
   typedef I value_type;
   //typedef I* pointer_type;
   
   //~Singleton() { delete instance_; }
   
   static value_type & instance();
   
 private:  
   Singleton();
   Singleton& operator=(const Singleton &);
   //static pointer_type instance_;
 };
}
#endif

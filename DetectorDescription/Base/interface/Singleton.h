#ifndef DDI_Singleton_h
#define DDI_Singleton_h

namespace DDI {
 template <class I> 
 class Singleton 
 {
 public:
   typedef I value_type;

   static value_type & instance();
   
 private:  
   Singleton();
   Singleton& operator=(const Singleton &);
 };
}
#endif

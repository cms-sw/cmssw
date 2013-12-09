#ifndef FWCore_Utilites_thread_safe_macros_h 
#define FWCore_Utilites_thread_safe_macros_h 
#ifndef __GCCXML__
#define CMS_THREAD_SAFE [[cms::thread_safe]]
#define CMS_THREAD_GUARD(_var_) [[cms::thread_guard("#_var_")]]
#else 
#define CMS_THREAD_SAFE
#define CMS_THREAD_GUARD(_var_)
#endif
#endif

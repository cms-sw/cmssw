#ifndef FWCore_Utilites_thread_safe_macros_h
#define FWCore_Utilites_thread_safe_macros_h
#if !defined __ROOTCLING__ && !defined __INTEL_COMPILER && !defined __NVCC__
#define CMS_THREAD_SAFE [[cms::thread_safe]]
#define CMS_SA_ALLOW [[cms::sa_allow]]
#define CMS_THREAD_GUARD(_var_) [[cms::thread_guard(#_var_)]]
#else
#define CMS_THREAD_SAFE
#define CMS_SA_ALLOW
#define CMS_THREAD_GUARD(_var_)
#endif
#endif

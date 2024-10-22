#ifndef FWCore_Utilites_deprecated_macro_h
#define FWCore_Utilites_deprecated_macro_h
#if !defined USE_CMS_DEPRECATED
#define CMS_DEPRECATED
#else
#define CMS_DEPRECATED [[deprecated]]
#endif
#endif

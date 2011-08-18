#ifndef FWCore_Utilities_typedefs_h
#define FWCore_Utilities_typedefs_h

// typedefs to provide platform independent types for members of ROOT persistent classes.
// To support class versioning for schema evolution, the typedef must resolve
// to the same built-in C++ type on all supported platforms.
// int64_t, uint64_t cannot be used, because they resolve to long on some platforms and long long on others.
// For consistency, we don't use int32_t or uint32_t, either.

typedef int cms_int32_t;
typedef unsigned int cms_uint32_t;
typedef long long cms_int64_t;
typedef unsigned long long cms_uint64_t;

#endif 

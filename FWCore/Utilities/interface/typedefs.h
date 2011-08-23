#ifndef FWCore_Utilities_typedefs_h
#define FWCore_Utilities_typedefs_h

// typedefs to provide platform independent types for members of ROOT persistent classes.
// To support class versioning for schema evolution, the typedef must resolve
// to the same built-in C++ type on all supported platforms.
// int64_t, uint64_t cannot be used, because they resolve to long on some platforms and long long on others.
// For consistency, we don't use int32_t, uint32_t, int16_t, uint16_t, int8_t or uint8_t, either.

typedef signed char cms_int8_t;
typedef unsigned char cms_uint8_t;
typedef short cms_int16_t;
typedef unsigned short cms_uint16_t;
typedef int cms_int32_t;
typedef unsigned int cms_uint32_t;
typedef long long cms_int64_t;
typedef unsigned long long cms_uint64_t;

#endif 

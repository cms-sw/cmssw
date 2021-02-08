#ifndef DIP_STDTYPES_H_INCLUDED
#define DIP_STDTYPES_H_INCLUDED

#include <string>
#include <iostream>

#ifdef WIN32
typedef __int64		superlong;		// for millisecond timestamps
#else
#include <sys/types.h> 
typedef __int64_t	superlong;
#endif


typedef bool DipBool;			//On 8bits
typedef unsigned char DipByte;	//On 8bits
typedef short DipShort;			//On 16bits
typedef int DipInt;				//On 32bits
typedef superlong DipLong;		//On 64bits
typedef float DipFloat;			//On 32bits
typedef double DipDouble;		//On 64bits

#endif //DIP_STDTYPES_H_INCLUDED


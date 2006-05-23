#ifndef _MsgTools_h
#define _MsgTools_h

#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <iterator>
#include <algorithm>
#include <cassert>
#include <cstdlib>

using namespace std;

// could just use the c99 names here from stdint.h
typedef unsigned char uint8;
typedef unsigned short uint16; 
typedef unsigned int  uint32;
typedef unsigned long long uint64;
typedef unsigned char char_uint32[sizeof(uint32)];
typedef unsigned char char_uint16[sizeof(uint16)];
typedef std::vector<std::string> Strings;

inline uint32 convert32(char_uint32 v)
{ 
  // first four bytes are code,  LSB first
  unsigned int a=v[0], b=v[1], c=v[2], d=v[3];
  a|=(b<<8)|(c<<16)|(d<<24);
  return a;
}

inline uint16 convert16(char_uint16 v)
{ 
  // first four bytes are code,  LSB first
  unsigned int a=v[0], b=v[1];
  a|=(b<<8);
  return a;
}

inline void convert(uint32 i, char_uint32 v)
{
  v[0]=i&0xff;
  v[1]=(i>>8)&0xff;
  v[2]=(i>>16)&0xff;
  v[3]=(i>>24)&0xff;
}

inline void convert(uint16 i, char_uint16 v)
{
  v[0]=i&0xff;
  v[1]=(i>>8)&0xff;
  v[2]=(i>>16)&0xff;
  v[3]=(i>>24)&0xff;
}

#endif


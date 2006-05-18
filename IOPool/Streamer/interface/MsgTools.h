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

/*****
void dumpInit(uint8* buf, uint32 bufsize)
{ 
  InitMsgView view(buf,bufsize);
 
  cout
    << "code = " << view.code() << ", "
    << "size = " << view.size() << "\n"
    << "run = " << view.run() << ", "
    << "proto = " << view.protocolVersion() << "\n"
    << "release = " << view.releaseTag() << "\n";
    
  uint8 vpset[17];
  view.pset(vpset); 
  vpset[16]='\0';
  Strings vhltnames,vl1names;
  view.hltTriggerNames(vhltnames);
  view.l1TriggerNames(vl1names);

  cout << "pset = " << vpset << "\n";
  cout << "\nHLT names = \n";
  copy(vhltnames.begin(),vhltnames.end(),ostream_iterator<string>(cout,"\n"));
  cout << "\nL1 names = \n";
  copy(vl1names.begin(),vl1names.end(),ostream_iterator<string>(cout,"\n"));
  cout << "\n";
  
  cout << "desc len = " << view.descLength() << "\n";
  const uint8* pos = view.descData();
  copy(pos,pos+view.descLength(),ostream_iterator<uint8>(cout,""));
  cout << "\n";
}

void dumpEvent(uint8* buf,uint32 bufsize,uint32 hltsize,uint32 l1size)
{
  EventMsgView eview(buf,bufsize,hltsize,l1size);

  cout << "----------------------\n";
  cout << "code=" << eview.code() << "\n"
       << "size=" << eview.size() << "\n"
       << "run=" << eview.run() << "\n"
       << "event=" << eview.event() << "\n"
       << "lumi=" << eview.lumi() << "\n"
       << "reserved=" << eview.reserved() << "\n"
       << "event length=" << eview.eventLength() << "\n";

  std::vector<bool> l1_out;
  uint8 hlt_out[10];
  eview.l1TriggerBits(l1_out);
  eview.hltTriggerBits(hlt_out);

  cout << "\nl1 size= " << l1_out.size() << " l1 bits=\n";
  copy(l1_out.begin(),l1_out.end(),ostream_iterator<bool>(cout," "));

  cout << "\nhlt bits=\n(";
  copy(&hlt_out[0],&hlt_out[0]+hltsize/4,ostream_iterator<char>(cout,""));
  cout << ")\n";

  const uint8* edata = eview.eventData();
  cout << "\nevent data=\n(";
  copy(&edata[0],&edata[0]+eview.eventLength(),
       ostream_iterator<char>(cout,""));
  cout << ")\n";

}
******/
#endif


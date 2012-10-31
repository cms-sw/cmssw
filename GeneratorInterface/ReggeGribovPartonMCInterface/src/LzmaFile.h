#ifndef _include_LzmaFile_h_
#define _include_LzmaFile_h_

#include "Types.h"
#include "7zFile.h"
#include "LzmaDec.h"

#include <string>
#include <queue>

#include "LzmaDec.h"
#include "Alloc.h"
#include "Types.h"
#include "7zFile.h"

#include <sstream>
#include <string>
#include <iostream>
#include <queue>
using namespace std;



#define IN_BUF_SIZE (1 << 16)
#define OUT_BUF_SIZE (1 << 16)

struct LzmaFile {

  LzmaFile();

  SRes Open(const std::string& fileName);
  SRes Close();
  SRes DecodeAll();
  SRes DecodeBuffer();
   //SRes DecodeArray(double* data, const int length);
  SRes ReadNextNumber(double& data);
  SRes FillArray(double* data, const int length);
  
  CFileSeqInStream inStream;
  int res;
  CLzmaDec state;
  
  // std::ostringstream fStrNumber;

  UInt64 unpackSize;
  
  Byte inBuf[IN_BUF_SIZE];
  Byte outBuf[OUT_BUF_SIZE];

  std::queue<double> fStorage;

  size_t inPos;
  size_t inSize;
  size_t outPos;

  // string converting
  bool fStartNumber;

  bool fReadSign;
  bool fReadMantisseR;
  bool fReadMantisseF;
  bool fReadExponentSign;
  bool fReadExponent;
  
  bool fNegative;
  bool fExponentNegative;
      
  double fMantisseR;
  double fMantisseF;
  int fMantisseFcount;
  int fExponent;

};



#endif

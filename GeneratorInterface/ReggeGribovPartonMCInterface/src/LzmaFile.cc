#include "LzmaFile.h"

#include "LzmaDec.h"
#include "Alloc.h"
#include "Types.h"
#include "7zFile.h"

//#include <sstream>
//#include <string>
#include <cmath>
#include <iostream>
#include <queue>
#include <cstdlib>
using namespace std;

const char *kCantReadMessage = "Can not read input file";
const char *kCantWriteMessage = "Can not write output file";
const char *kCantAllocateMessage = "Can not allocate memory";
const char *kDataErrorMessage = "Data error";

static void *SzAlloc(void *, size_t size) { return MyAlloc(size); }
static void SzFree(void *, void *address) { MyFree(address); }
static ISzAlloc g_Alloc = {SzAlloc, SzFree};

LzmaFile::LzmaFile() {
  // fStorage.reserve(10000);
  fStartNumber = false;

  fReadSign = true;
  fReadMantisseR = true;
  fReadMantisseF = false;
  fReadExponentSign = false;
  fReadExponent = false;

  fNegative = false;
  fExponentNegative = false;

  fMantisseR = 0;
  fMantisseF = 0;
  fMantisseFcount = 0;
  fExponent = 0;
}

SRes LzmaFile::Open(const string &fileName) {
  //fStrNumber.str("");
  //fStrNumber.clear();

  FileSeqInStream_CreateVTable(&inStream);
  File_Construct(&inStream.file);

  if (InFile_Open(&inStream.file, fileName.c_str()) != 0) {
    cout << "Cannot open input file: " << fileName << endl;
    cout << "First use: \n\t \'lzma --best " << fileName.substr(0, fileName.rfind(".lzma")) << "\'"
         << " to create it. " << endl;
    exit(1);
  }

  ISeqInStream *stream = &inStream.s;

  /* Read and parse header */
  /* header: 5 bytes of LZMA properties and 8 bytes of uncompressed size */
  unsigned char header[LZMA_PROPS_SIZE + 8];
  RINOK(SeqInStream_Read(stream, header, sizeof(header)));

  unpackSize = 0;
  int i = 0;
  for (i = 0; i < 8; i++)
    unpackSize += (UInt64)header[LZMA_PROPS_SIZE + i] << (i * 8);

  LzmaDec_Construct(&state);
  RINOK(LzmaDec_Allocate(&state, header, LZMA_PROPS_SIZE, &g_Alloc));
  LzmaDec_Init(&state);

  inPos = 0;
  inSize = 0;
  outPos = 0;
  return SZ_OK;
}

SRes LzmaFile::ReadNextNumber(double &data) {
  if (fStorage.empty()) {
    const int ret = DecodeBuffer();
    if (ret != SZ_OK) {
      cout << "Error in ReadNextNumber  ret=" << ret << endl;
      return SZ_ERROR_DATA;
    }
  }

  data = fStorage.front();
  fStorage.pop();
  return SZ_OK;
}

SRes LzmaFile::FillArray(double *data, const int length) {
  for (int i = 0; i < length; ++i) {
    if (fStorage.empty()) {
      const int ret = DecodeBuffer();
      if (ret != SZ_OK) {
        cout << "Error in FillArray i=" << i << " ret=" << ret << endl;
        return SZ_ERROR_DATA;
      }
    }

    data[i] = fStorage.front();
    fStorage.pop();
  }

  return SZ_OK;
}

/*
double 
LzmaFile::strToDouble(const char& p) {

  // init
  fR = 0;  
  fNegative = false;
  
  if (*p == '-') {
    neg = true;
    ++p;
  }
  while (*p >= '0' && *p <= '9') {
    r = (r*10.0) + (*p - '0');
    ++p;
  }
  if (*p == '.') {
    double f = 0.0;
    int n = 0;
    ++p;
    while (*p >= '0' && *p <= '9') {
      f = (f*10.0) + (*p - '0');
      ++p;
      ++n;
    }
    r += f / std::pow(10.0, n);
  }
  if (neg) {
    r = -r;
  }
  return r;
}
*/

SRes LzmaFile::DecodeBuffer() {
  ISeqInStream *stream = &inStream.s;

  const int thereIsSize = (unpackSize != (UInt64)(Int64)-1);

  if (inPos == inSize) {
    inSize = IN_BUF_SIZE;
    RINOK(stream->Read(stream, inBuf, &inSize));
    inPos = 0;
  }

  SizeT inProcessed = inSize - inPos;
  SizeT outProcessed = OUT_BUF_SIZE - outPos;
  ELzmaFinishMode finishMode = LZMA_FINISH_ANY;
  ELzmaStatus status;

  if (thereIsSize && outProcessed > unpackSize) {
    outProcessed = (SizeT)unpackSize;
    finishMode = LZMA_FINISH_END;
  }

  SRes res = LzmaDec_DecodeToBuf(&state, outBuf, &outProcessed, inBuf + inPos, &inProcessed, finishMode, &status);
  inPos += inProcessed;
  unpackSize -= outProcessed;

  const char *strBuf = (const char *)outBuf;

  int countC = 0;
  do {
    if (countC >= int(outProcessed)) {
      // cout << " countC=" << countC
      // 	   << " outProcessed=" << outProcessed
      // 	   << endl;
      break;
    }

    const char &C = strBuf[countC];
    countC++;

    //cout << "\'" << C << "\'" << endl;

    if (C == ' ' || C == '\n') {  // END OF NUMBER

      if (!fStartNumber)
        continue;

      //istringstream strToNum(fStrNumber.str().c_str());
      //double number = atof(fStrNumber.str().c_str());
      //strToNum >> number;

      const double number = (fNegative ? -1 : 1) * (fMantisseR + fMantisseF / pow(10, fMantisseFcount)) *
                            pow(10, (fExponentNegative ? -1 : 1) * fExponent);
      //cout << " number=" << number << endl;

      fStorage.push(number);

      fStartNumber = false;

      fReadSign = true;
      fReadMantisseR = true;
      fReadMantisseF = false;
      fReadExponentSign = false;
      fReadExponent = false;

      fNegative = false;
      fExponentNegative = false;

      fMantisseR = 0;
      fMantisseF = 0;
      fMantisseFcount = 0;
      fExponent = 0;

      continue;
    }

    fStartNumber = true;
    const int num = C - '0';
    if (num >= 0 && num <= 9) {
      if (fReadMantisseR) {
        fReadSign = false;
        fMantisseR = fMantisseR * 10 + num;
      } else if (fReadMantisseF) {
        fReadSign = false;
        fMantisseF = fMantisseF * 10 + num;
        ++fMantisseFcount;
      } else if (fReadExponent) {
        fReadExponentSign = false;
        fExponent = fExponent * 10 + num;
      }

    } else {
      switch (C) {
        case '-': {
          if (fReadSign) {
            fNegative = true;
            fReadSign = false;
            fReadMantisseR = true;
          } else if (fReadExponentSign) {
            fExponentNegative = true;
            fReadExponentSign = false;
            fReadExponent = true;
          } else {
            cout << "LzmaFile: found \'" << C << "\' at wrong position. " << endl;
            exit(10);
          }
        } break;
        case '.':
          if (!fReadMantisseR) {
            cout << "LzmaFile: found \'" << C << "\' at wrong position. " << endl;
            exit(10);
          }
          fReadMantisseR = false;
          fReadMantisseF = true;
          break;
        case 'e':
        case 'E':
        case 'D':
        case 'd':
          if (!fReadMantisseR || !fReadMantisseF) {
            fReadMantisseR = false;
            fReadMantisseF = false;
            fReadExponentSign = true;
            fReadExponent = true;
          }
          break;
        default:
          cout << "LzmaFile: found \'" << C << "\' at wrong position. " << endl;
          exit(10);
          break;
      }
    }

  } while (true);

  //strBuf.str("");
  //strBuf.clear();

  /*    
  if (!strNumber.str().empty()) {
    cout << "NACHZUEGLER" << endl;
    istringstream strToNum(strNumber.str());
    double number;
    strToNum >> number;
    fStorage.push(number);
  }
  */

  if (res != SZ_OK || (thereIsSize && unpackSize == 0))
    return res;

  if (inProcessed == 0 && outProcessed == 0) {
    if (thereIsSize || status != LZMA_STATUS_FINISHED_WITH_MARK)
      return SZ_ERROR_DATA;
    return res;
  }

  return SZ_OK;
}

SRes LzmaFile::DecodeAll() {
  ISeqInStream *stream = &inStream.s;

  int thereIsSize = (unpackSize != (UInt64)(Int64)-1);

  for (;;) {
    if (inPos == inSize) {
      inSize = IN_BUF_SIZE;
      RINOK(stream->Read(stream, inBuf, &inSize));
      inPos = 0;
    }

    SizeT inProcessed = inSize - inPos;
    SizeT outProcessed = OUT_BUF_SIZE - outPos;
    ELzmaFinishMode finishMode = LZMA_FINISH_ANY;
    ELzmaStatus status;

    if (thereIsSize && outProcessed > unpackSize) {
      outProcessed = (SizeT)unpackSize;
      finishMode = LZMA_FINISH_END;
    }

    SRes res = LzmaDec_DecodeToBuf(&state, outBuf, &outProcessed, inBuf + inPos, &inProcessed, finishMode, &status);
    inPos += inProcessed;
    unpackSize -= outProcessed;

    unsigned int k = 0;
    for (k = 0; k < outProcessed; ++k) {
      printf("%c", outBuf[k]);
    }

    if (res != SZ_OK || (thereIsSize && unpackSize == 0))
      return res;

    if (inProcessed == 0 && outProcessed == 0) {
      if (thereIsSize || status != LZMA_STATUS_FINISHED_WITH_MARK)
        return SZ_ERROR_DATA;
      return res;
    }

  }  // for loop

  return 0;
}

SRes LzmaFile::Close() {
  LzmaDec_Free(&state, &g_Alloc);
  res = File_Close(&inStream.file);
  return res;
}

#include "LzmaFile.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>







int 
main(int numArgs, const char *args[])
{
  int res;

  if (numArgs != 2)
  {
    //PrintHelp(rs);
    return 0;
  }


  {
    size_t t4 = sizeof(UInt32);
    size_t t8 = sizeof(UInt64);
    if (t4 != 4 || t8 != 8) {
      //return PrintError(rs, "Incorrect UInt32 or UInt64");
    }
  }

  LzmaFile lzma;
  res = lzma.Open(args[1]);
  res = lzma.DecodeAll();
  res = lzma.Close();

  if (res != SZ_OK)
  {
    if (res == SZ_ERROR_MEM) {
      return 0;//PrintError(rs, kCantAllocateMessage);
    } else if (res == SZ_ERROR_DATA) {
      return 0;//PrintError(rs, kDataErrorMessage);
    } else if (res == SZ_ERROR_WRITE) {
      return 0;//PrintError(rs, kCantWriteMessage);
    } else if (res == SZ_ERROR_READ) {
      return 0;//PrintError(rs, kCantReadMessage);
    }
    return 0;//PrintErrorNumber(rs, res);
  }
  return 0;
}

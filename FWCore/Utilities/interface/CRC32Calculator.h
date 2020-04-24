#ifndef FWCore_Utilities_CRC32Calculator_h
#define FWCore_Utilities_CRC32Calculator_h

/*
Code to calculate a CRC32 checksum on a string.  This code is based
on code copied from the web in the public domain.  The code was modified
quite a bit to provide the interface we need, remove extraneous code
related to NAACR records, convert from C to C++, and use a
boost type for the 32 unsigned integers, but the essential features
of the CRC calculation are the same.  The array values, constants,
and algorithmic steps are identical.  The comments in the header
from the original code follow below to attribute the source.
*/

/* crc32.c

   C implementation of CRC-32 checksums for NAACCR records.  Code is based
   upon and utilizes algorithm published by Ross Williams.

   This file contains:
      CRC lookup table
      function CalcCRC32 for calculating CRC-32 checksum
      function AssignCRC32 for assigning CRC-32 in NAACCR record
      function CheckCRC32 for checking CRC-32 in NAACCR record

   Provided by:
      Eric Durbin
      Kentucky Cancer Registry
      University of Kentucky
      October 14, 1998

   Status:
      Public Domain
*/

/*****************************************************************/
/*                                                               */
/* CRC LOOKUP TABLE                                              */
/* ================                                              */
/* The following CRC lookup table was generated automagically    */
/* by the Rocksoft^tm Model CRC Algorithm Table Generation       */
/* Program V1.0 using the following model parameters:            */
/*                                                               */
/*    Width   : 4 bytes.                                         */
/*    Poly    : 0x04C11DB7L                                      */
/*    Reverse : TRUE.                                            */
/*                                                               */
/* For more information on the Rocksoft^tm Model CRC Algorithm,  */
/* see the document titled "A Painless Guide to CRC Error        */
/* Detection Algorithms" by Ross Williams                        */
/* (ross@guest.adelaide.edu.au.). This document is likely to be  */
/* in the FTP archive "ftp.adelaide.edu.au/pub/rocksoft".        */
/*                                                               */
/*****************************************************************/

#include <cstdint>

#include <string>

namespace cms {

  class CRC32Calculator {

  public:

    CRC32Calculator(std::string const& message);

    std::uint32_t checksum() { return checksum_; }

  private:

    std::uint32_t checksum_;
  };
}
#endif

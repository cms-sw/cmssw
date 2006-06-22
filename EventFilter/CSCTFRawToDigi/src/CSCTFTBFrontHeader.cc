// -*- C++ -*-
//
// Package:     CSCTFTBRawFormat
// Module:      CSCTFTBFrontHeader
// 
// Description: Contains front block header information
//              (see SP Data format)
//
// Implementation:
//     <Notes on implementation>
//
// Author:      Lindsey Gray
// Created:     13.1.2005
//
// $Id: CSCTFTBFrontHeader.cc,v 1.1 2006/02/22 23:16:42 lgray Exp $

// System include files

// Package include files
#include "EventFilter/CSCTFRawToDigi/src/CSCTFTBFrontHeader.h"

// External package include files

// STL classes

// Constants, enums and typedefs

// CVS-based strings (Id and Tag with which file was checked out)
static const char* const kIdString  = "$Id: CSCTFTBFrontHeader.cc,v 1.1 2006/02/22 23:16:42 lgray Exp $";
static const char* const kTagString = "$Name:  $";

// Static data member definitions

// Constructors and destructor

CSCTFTBFrontHeader::CSCTFTBFrontHeader():
  zero1_(0),
  zero2_(0)
{}

CSCTFTBFrontHeader::CSCTFTBFrontHeader(const CSCTFTBFrontHeader& parent)
{
  memcpy(this,&parent,size()*sizeof(unsigned short));
}

CSCTFTBFrontHeader::~CSCTFTBFrontHeader()
{}

/// Valid Pattern for given link
unsigned CSCTFTBFrontHeader::getVPBit(unsigned int frontFPGA, unsigned int link) const  
{
  frontFPGA =  (frontFPGA >5 || frontFPGA < 1) ? 1 : frontFPGA;
  link =  (link >3 || link < 1) ? 1 : link;
  unsigned shift = ( (frontFPGA-1)*3 + (link-1) );
  return (vld_patrn_ & (1<<shift)) >> shift;
}

/// Synch Error for given link
unsigned CSCTFTBFrontHeader::getSEBit(unsigned int frontFPGA, unsigned int link) const  
{
  frontFPGA =  (frontFPGA >5 || frontFPGA < 1) ? 1 : frontFPGA;
  link =  (link >3 || link < 1) ? 1 : link;
  unsigned shift = ( (frontFPGA-1)*3 + (link-1) );
  return (synch_err_ & (1<<shift)) >> shift;
}

std::ostream & operator<<(std::ostream & stream, const CSCTFTBFrontHeader & hdr) 
{
  if(hdr.validPattern())
    {
      stream << "\tFront Header Data:\n";
      stream << "\t Valid Pattern Bits : " << std::hex << hdr.validPattern()
	     << std::endl;
      stream << "\t Synch Error Bits   : " << std::hex << hdr.synchError()
	     <<std::endl;
    }
  return stream;
}

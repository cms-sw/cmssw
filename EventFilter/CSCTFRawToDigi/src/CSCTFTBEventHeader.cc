// -*- C++ -*-
//
// Package:     CSCTFTBRawFormat
// Module:      CSCTFTBEventHeader
// 
// Description: SP VME Header class
//  Unpacks and stores CSCTFTBSPVMEHeader data in member variables of CSCTFTBSPVMEHeader.h;
//  rewritten to use bit fields.
//
// Implementation:
//     <Notes on implementation>
//
// Author:      Darin Acosta / Holger Stoeck / Lindsey Gray
// Created:     13.1.2005
//
// $Id: CSCTFTBEventHeader.cc,v 1.1 2006/06/22 00:34:18 lgray Exp $
//
// Revision History
// $Log: CSCTFTBEventHeader.cc,v $
// Revision 1.1  2006/06/22 00:34:18  lgray
// Moved all data format classes here. Removed old Packages from nightly
//
// Revision 1.1  2006/02/22 23:16:42  lgray
// First commit of test beam data format from UF
//
// Revision 1.2  2005/05/13 08:29:59  lgray
// Another bug fix.
//
// Revision 1.1  2005/02/14 21:01:32  lgray
// First Commit from UF
//
// Revision 1.15  2004/05/28 00:24:58  tfcvs
// DEA: a working version of code for 4 chambers!
//
// Revision 1.14  2004/05/18 21:53:42  tfcvs
// DEA: some print out
//
// Revision 1.13  2004/05/18 21:05:06  tfcvs
// DEA: close to working
//
// Revision 1.12  2004/05/18 08:00:35  tfcvs
// DEA: touch base
//
// Revision 1.11  2004/05/17 15:19:49  tfcvs
// DEA: expand header for TB 2004
//
// Revision 1.10  2004/05/16 07:43:49  tfcvs
// DEA: TB2003 version working with new software
//
// Revision 1.9  2003/09/19 20:22:55  tfcvs
// latest
//
// Revision 1.8  2003/08/27 22:09:14  tfcvs
// Added pretty-print  -Rick
//
// Revision 1.7  2003/05/25 10:13:02  tfcvs
// first working version -DEA
//
// Revision 1.6  2003/05/20 22:13:06  tfcvs
// HS - Added Darin's changes
//
// Revision 1.5  2003/05/19 23:23:12  tfcvs
// HS - Commit after some changes
//
// Revision 1.3  2003/05/19 15:47:18  tfcvs
// HS - Some cleanup
//
// Revision 1.2  2003/05/15 23:58:40  tfcvs
// HS - Some cosmetics
//
// 
//

// System include files
#include <iostream>
#include <stdio.h>

// Package include files
#include "EventFilter/CSCTFRawToDigi/src/CSCTFTBEventHeader.h"

// External package include files
// #include "DMBHeader.h"
// #include "ALCTHeader.h"

// STL classes

// Constants, enums and typedefs

// CVS-based strings (Id and Tag with which file was checked out)
static const char* const kIdString  = "$Id: CSCTFTBEventHeader.cc,v 1.1 2006/06/22 00:34:18 lgray Exp $";
static const char* const kTagString = "$Name:  $";

// Static data member definitions

// Constructors and destructor

CSCTFTBEventHeader::CSCTFTBEventHeader()
{}

CSCTFTBEventHeader::CSCTFTBEventHeader(const CSCTFTBEventHeader& parent)
{
  memcpy(reinterpret_cast<unsigned*>(this),&parent,size()*sizeof(unsigned short));
}

CSCTFTBEventHeader::~CSCTFTBEventHeader()
{}

unsigned int CSCTFTBEventHeader::getNumActiveFront() const
{
  unsigned int num = 0;

  for(int i=0;i<5;i++)
    {
      if(act_ffpga_&(1<<i)) num++;
    }
  return num;
}

bool CSCTFTBEventHeader::check() const
{
  return (key1_ == 0xf && key2_ == 0xf && key3_ == 0xf && key4_ == 0xf);
}

std::ostream & operator<<(std::ostream & stream, const CSCTFTBEventHeader & header) {
  stream << std::dec <<"Event Header for L1A " << header.getLvl1num()<<" :\n"
         << " BX " << header.getBXnum() << std::endl
         << " " << header.numMPC() << " MPC(s)" 
         << " with " << header.numLinks() << " links per MPC.\n"
         << " There are " << header.numBX() << " BX in this event." << std::endl;
  if(header.getZeroSupp()) stream << " Zero Suppression is active.\n";
  if(header.getActiveDT()) stream << " DT is active.\n";
  if(header.getPtLutSpy()) 
    stream << " PT LUT data present on link " << header.getPtLutSpy()<<".\n";
  if(header.getActiveFront())
    {
      stream << " There are "<<header.getNumActiveFront()<<" active front FPGAs." <<std::endl;
    }
  stream<<std::hex;
  return stream;
}

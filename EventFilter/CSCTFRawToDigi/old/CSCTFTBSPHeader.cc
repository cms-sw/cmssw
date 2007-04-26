// -*- C++ -*-
//
// Package:     CSCTFTBRawFormat
// Module:      CSCTFTBSPHeader
// 
// Description: SP Event Data class
//
// Implementation:
//     <Notes on implementation>
//
// Author:      Lindsey Gray
// Created:     13.1.2005
//
// $Id: CSCTFTBSPHeader.cc,v 1.2 2006/06/22 14:46:05 lgray Exp $
//
// Revision History
// $Log: CSCTFTBSPHeader.cc,v $
// Revision 1.2  2006/06/22 14:46:05  lgray
// Forced commit of all code
//
// Revision 1.1  2006/06/22 00:34:18  lgray
// Moved all data format classes here. Removed old Packages from nightly
//
// Revision 1.1  2006/02/22 23:16:42  lgray
// First commit of test beam data format from UF
//
// Revision 1.1  2005/02/14 21:01:32  lgray
// First Commit from UF
//
// Revision 1.10  2004/05/28 00:24:58  tfcvs
// DEA: a working version of code for 4 chambers!
//
// Revision 1.9  2004/05/18 21:53:42  tfcvs
// DEA: some print out
//
// Revision 1.8  2004/05/18 15:00:25  tfcvs
// DEA: close to new SP data format
//
// Revision 1.7  2004/05/18 11:37:46  tfcvs
// DEA: touch base
//
// Revision 1.6  2004/05/18 09:45:07  tfcvs
// DEA: touch base
//
// Revision 1.5  2004/05/18 08:00:34  tfcvs
// DEA: touch base
//
// Revision 1.4  2004/05/16 07:43:49  tfcvs
// DEA: TB2003 version working with new software
//
// Revision 1.3  2003/09/19 20:22:55  tfcvs
// latest
//
// Revision 1.2  2003/08/27 22:08:59  tfcvs
// Added pretty-print  -Rick
//
// Revision 1.1  2003/05/25 10:13:02  tfcvs
// first working version -DEA
//
// Revision 1.7  2003/05/22 17:27:20  tfcvs
// HS - Some more minor changes
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

// Package include files
#include "EventFilter/CSCTFRawToDigi/src/CSCTFTBSPHeader.h"

// External package include files
// #include "CSCEventData.h"

// STL classes

// Constants, enums and typedefs

// CVS-based strings (Id and Tag with which file was checked out)
static const char* const kIdString  = "$Id: CSCTFTBSPHeader.cc,v 1.2 2006/06/22 14:46:05 lgray Exp $";
static const char* const kTagString = "$Name:  $";

// Static data member definitions

// Constructors and destructor
CSCTFTBSPHeader::CSCTFTBSPHeader()
{}

CSCTFTBSPHeader::CSCTFTBSPHeader(const CSCTFTBSPHeader &parent)
{
  memcpy(this,&parent,size()*sizeof(unsigned short));
}

CSCTFTBSPHeader::~CSCTFTBSPHeader()
{
}

CSCTFTBSPHeader CSCTFTBSPHeader::operator=(const CSCTFTBSPHeader & parent)
{
  memcpy(this,&parent,size()*sizeof(unsigned short));
  return *this;
}

// Member Functions

/// Synch Error for given link
unsigned int CSCTFTBSPHeader::getSEBit(unsigned int frontFPGA, unsigned int link) 
const  
{
  frontFPGA =  (frontFPGA >5 || frontFPGA < 1) ? 1 : frontFPGA;
  link =  (link >3 || link < 1) ? 1 : link;
  unsigned shift = ( (frontFPGA-1)*3 + (link-1) );
  return (synch_err_ & (1<<shift)) >> shift;
}

/// Track Mode (i.e. quality)  for given track
unsigned int CSCTFTBSPHeader::getTrackMode(unsigned int trk) const
{
  switch(trk)
    {
    case 1:
      return mode1_;
      break;
    case 2:
      return mode2_;
      break;
    case 3:
      return mode3_;
      break;
    default:
      return mode1_;
      break;
    }
}

/// Valid Pattern for given link
unsigned int CSCTFTBSPHeader::getVPDTBit(unsigned int link) const  
{
  switch(link)
    {
    case 1:
      return MB1A_flag_;
      break;
    case 2:
      return MB1D_flag_;
      break;
    default:
      return MB1A_flag_;
      break;
    }
}


std::ostream & operator<<(std::ostream & stream, const CSCTFTBSPHeader & bx) 
{
  if(bx.synchError()||bx.getTrackMode(1)||bx.getTrackMode(2)
     ||bx.getTrackMode(3)||bx.getVPDTBit(1) || bx.getVPDTBit(2))
    {
      stream <<"\tSP Header Data:"<< std::endl;
      stream <<"\t Synch Error Bits   : "<< std::hex << bx.synchError()<< std::dec <<std::endl;
      if(bx.getTrackMode(1)||bx.getTrackMode(2)||bx.getTrackMode(3))
	{
	  stream <<"\t Track Modes:\n";
	  if(bx.getTrackMode(1)) stream<<"\t\tTrack 1: "<< bx.getTrackMode(1)<< "\n";
	  if(bx.getTrackMode(2)) stream<<"\t\tTrack 2: "<< bx.getTrackMode(2)<< "\n";
	  if(bx.getTrackMode(3)) stream<<"\t\tTrack 3: "<< bx.getTrackMode(3)<< "\n";
	}
      if(bx.getVPDTBit(1) || bx.getVPDTBit(2))
	{
	  stream <<"\t DT Valid Patterns:\n";
	  if(bx.getVPDTBit(1)) stream <<"\t\tValid DT pattern in link 1!\n";
	  if(bx.getVPDTBit(2)) stream <<"\t\tValid DT pattern in link 2!\n";
	}
    }
  return stream;
}


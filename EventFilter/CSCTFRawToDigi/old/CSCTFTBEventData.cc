// -*- C++ -*-
//
// Package:     CSCTFTBRawFormat
// Module:      CSCTFTBEventData
// 
// Description: SR/SP Event Data class
//
// Implementation:
//     <Notes on implementation>
//
// Author:      Lindsey Gray
// Created:     13.1.2005
//
// $Id: CSCTFTBEventData.cc,v 1.2 2006/06/22 14:46:05 lgray Exp $
//
// Revision History
// $Log: CSCTFTBEventData.cc,v $
// Revision 1.2  2006/06/22 14:46:05  lgray
// Forced commit of all code
//
// Revision 1.1  2006/06/22 00:34:18  lgray
// Moved all data format classes here. Removed old Packages from nightly
//
// Revision 1.2  2006/02/27 10:43:57  lgray
// Print changes.
//
// Revision 1.1  2006/02/22 23:16:42  lgray
// First commit of test beam data format from UF
//
// Revision 1.6  2005/06/22 22:03:27  lgray
// update
//
// Revision 1.5  2005/05/10 21:57:22  lgray
// Bugfixes, stability issues fixed
//
// Revision 1.4  2005/03/03 19:52:27  lgray
// Minor changes
//
// Revision 1.3  2005/03/03 18:14:48  lgray
// Added ability to pack data back into raw form. Added test program for this as well.
//
// Revision 1.2  2005/02/15 17:29:45  lgray
// Getting ready for SP DDU
//
// Revision 1.1  2005/02/14 21:01:32  lgray
// First Commit from UF
//
// Revision 1.8  2004/11/30 02:59:27  tfcvs
// (LAG) Better way.
//
// Revision 1.7  2004/11/30 02:55:27  tfcvs
// (LAG) fixed small problem with CDF headers.
//
// Revision 1.6  2004/11/29 23:01:03  tfcvs
// (LAG) Added static bool to account for data with CDF header/trailer
//
// Revision 1.5  2004/05/18 15:00:25  tfcvs
// DEA: close to new SP data format
//
// Revision 1.4  2004/05/18 11:37:46  tfcvs
// DEA: touch base
//
// Revision 1.3  2004/05/18 08:00:35  tfcvs
// DEA: touch base
//
// Revision 1.2  2004/05/17 08:25:52  tfcvs
// DEA: switch to SR BX data
//
// Revision 1.1  2004/05/16 12:17:17  tfcvs
// DEA: call overall data block 
//
// Revision 1.12  2004/05/16 07:43:49  tfcvs
// DEA: TB2003 version working with new software
//
// Revision 1.11  2003/09/19 20:22:55  tfcvs
// latest
//
// Revision 1.10  2003/08/27 22:09:04  tfcvs
// Added pretty-print  -Rick
//
// Revision 1.9  2003/05/27 12:45:00  tfcvs
// add BXN and L1A # to SPEvent -DEA
//
// Revision 1.8  2003/05/25 10:13:02  tfcvs
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
#include <iostream>
#include <vector>

// Package include files
#include "EventFilter/CSCTFRawToDigi/src/CSCTFTBEventData.h"
#include "Utilities/Timing/interface/TimingReport.h"

// External package include files

// STL classes

// Constants, enums and typedefs

// CVS-based strings (Id and Tag with which file was checked out)
static const char* const kIdString  = "$Id: CSCTFTBEventData.cc,v 1.2 2006/06/22 14:46:05 lgray Exp $";
static const char* const kTagString = "$Name:  $";

// Static data member definitions
bool CSCTFTBEventData::hasCDF_ = false;
unsigned CSCTFTBEventData::trigBX_ = 0;
bool CSCTFTBEventData::onlyTrigBX_ = false;

// Constructors and destructor
CSCTFTBEventData::CSCTFTBEventData()
{
}

//
CSCTFTBEventData::CSCTFTBEventData(unsigned short *buf)
{
// Then unpack the rest of the data

  unpackData(buf);
}

CSCTFTBEventData::~CSCTFTBEventData()
{
  frontBlocks_.clear();
  spBlocks_.clear();
}


// Member Functions

int CSCTFTBEventData::unpackData(unsigned short *buf) 
{
//   cout << " CSCTFTBEventData_unpackData-INFO: Unpacking event data" << endl;
  unsigned short *inputbuf = buf;

  TimeMe t("CSCTFTBEventData::unpack"); 
  
  memcpy(&spHead_,buf,CSCTFTBEventHeader::size()*sizeof(unsigned short));
  
  buf += CSCTFTBEventHeader::size();

  frontBlocks_.clear();
  spBlocks_.clear();
  if (spHead_.numBX() > 0) {
     for (int bx = 1; bx<=spHead_.numBX(); bx++)
       {
	 CSCTFTBFrontBlock aFrontBlock(buf, bx,spHead_);
	 frontBlocks_.push_back(aFrontBlock);
	 buf+= aFrontBlock.size();	 
	 CSCTFTBSPBlock aSPBlock(buf, bx,spHead_);
         spBlocks_.push_back(aSPBlock);
	 buf+= aSPBlock.size();	
       }
   }
  
  size_ = buf - inputbuf;
  inputbuf = NULL;
  return 0;

}

void CSCTFTBEventData::setEventInformation(unsigned bxnum,unsigned lvl1num)
{
  spHead_.bunch_cntr_ = bxnum;
  spHead_.l1a_lsb_ = (lvl1num & 0xFFF);
  spHead_.l1a_msb_ = ((lvl1num>>12) & 0xFFF);
}

/* put back in when BitVector is back.
BitVector CSCTFTBEventData::pack()
{
  BitVector result;

  BitVector evhdr(reinterpret_cast<unsigned*>(&spHead_),spHead_.size()*sizeof(short)*8);
  result.assign(result.nBits(),evhdr.nBits(),evhdr);

  for(int i = 0; i< spHead_.numBX();i++)
    {
      BitVector front = frontBlocks_[i].pack();
      BitVector sp = spBlocks_[i].pack();
      result.assign(result.nBits(),front.nBits(),front);
      result.assign(result.nBits(),sp.nBits(),sp);
    }
  return result;
}
*/

std::ostream & operator<<(std::ostream & stream, const CSCTFTBEventData & data) {
  stream << data.eventHeader();
  std::vector<CSCTFTBFrontBlock> srBxData = data.frontData();
  std::vector<CSCTFTBSPBlock> spBxData = data.spData();
  for(unsigned bx = 0; bx < srBxData.size(); ++bx) {
    if((bx+1)==data.trigBX_ || !data.onlyTrigBX_)
      {
	stream << "Event data for bx: " << bx+1 << std::endl;
	stream << srBxData[bx];
	stream << spBxData[bx];
      }
  }
  stream << std::endl;
  return stream;
}

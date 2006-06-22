// -*- C++ -*-
//
// Package:     CSCTFTBFrontBlock
// Module:      CSCTFTBFrontBlock
// 
// Description: SP Event Data class
//
// Implementation:
//     <Notes on implementation>
//
// Author:      Lindsey Gray
// Created:     24.1.2005
//
// $Id: CSCTFTBFrontBlock.cc,v 1.1 2006/06/22 00:34:18 lgray Exp $
//
// Revision History
// $Log: CSCTFTBFrontBlock.cc,v $
// Revision 1.1  2006/06/22 00:34:18  lgray
// Moved all data format classes here. Removed old Packages from nightly
//
// Revision 1.6  2006/04/08 06:18:15  lgray
// Conform to change in CSCDigi
//
// Revision 1.5  2006/03/02 14:53:12  lgray
// Changed pretty print so I can log it.
//
// Revision 1.4  2006/02/27 10:43:57  lgray
// Print changes.
//
// Revision 1.3  2006/02/27 01:19:24  lgray
// Changes to print functions and test data file
//
// Revision 1.2  2006/02/26 23:34:34  lgray
// Adding tests for raw data.
//
// Revision 1.1  2006/02/22 23:16:42  lgray
// First commit of test beam data format from UF
//
// Revision 1.3  2005/06/22 22:03:27  lgray
// update
//
// Revision 1.2  2005/03/03 18:14:48  lgray
// Added ability to pack data back into raw form. Added test program for this as well.
//
// Revision 1.1  2005/02/14 21:01:32  lgray
// First Commit from UF
//
// Revision 1.4  2004/05/28 00:24:58  tfcvs
// DEA: a working version of code for 4 chambers!
//
// Revision 1.3  2004/05/18 21:53:42  tfcvs
// DEA: some print out
//
// Revision 1.2  2004/05/18 08:00:35  tfcvs
// DEA: touch base
//
// Revision 1.1  2004/05/17 08:25:52  tfcvs
// DEA: switch to SR BX data
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
#include <iostream>
#include <vector>
#include <string.h> //memcpy
// Package include files
#include "EventFilter/CSCTFRawToDigi/src/CSCTFTBEventHeader.h"
#include "EventFilter/CSCTFRawToDigi/src/CSCTFTBFrontBlock.h"
//#include "Utilities/GenUtil/interface/BitVector.h"

// External package include files
// #include "CSCEventData.h"

// STL classes

// Constants, enums and typedefs

// CVS-based strings (Id and Tag with which file was checked out)
static const char* const kIdString  = "$Id: CSCTFTBFrontBlock.cc,v 1.1 2006/06/22 00:34:18 lgray Exp $";
static const char* const kTagString = "$Name:  $";

// Static data member definitions

// Constructors and destructor
CSCTFTBFrontBlock::CSCTFTBFrontBlock()
{
  myBX_ = 0;
  size_ = 0;
  frontHeader_ = CSCTFTBFrontHeader();
}


//
CSCTFTBFrontBlock::CSCTFTBFrontBlock(unsigned short *buf, int bx,
				     const CSCTFTBEventHeader & thehdr)
{

  myBX_ = bx;
  size_ = 0;
/*
  cout << "SR Bx Data : " << endl;
  cout << " Front valid pattern = " << hex << validPattern_ << dec <<endl;
*/
  unpackData(buf, thehdr);
}

CSCTFTBFrontBlock::~CSCTFTBFrontBlock()
{
  std::vector<std::vector<CSCTFTBFrontData> >::iterator the_data;

  for(the_data = srdata_.begin(); the_data != srdata_.end() ; the_data++)
    {
      the_data->clear();
    }
  srdata_.clear();
}

// Member Functions

/// Accessor to link triplet of one MPC
std::vector<CSCTFTBFrontData> CSCTFTBFrontBlock::frontData(unsigned mpc ) const
{
   mpc -= 1;
   if(mpc < srdata_.size()) return srdata_[mpc];
   return std::vector<CSCTFTBFrontData>();
}

/// Accessor to one link's data
CSCTFTBFrontData CSCTFTBFrontBlock::frontData(unsigned mpc,unsigned link ) 
  const
{
  mpc -= 1;
  link -= 1; 
  if(srdata_[mpc].size() && (link < srdata_[mpc].size())) return srdata_[mpc][link];
  return CSCTFTBFrontData(mpc+1);
}

CSCCorrelatedLCTDigi CSCTFTBFrontBlock::frontDigiData(unsigned mpc, unsigned link) const
{
  mpc -= 1;
  link -= 1;
  if(srdata_[mpc].size() && (link < srdata_[mpc].size()))
    {
      CSCTFTBFrontData aFD = srdata_[mpc][link];
      return CSCCorrelatedLCTDigi(0, frontHeader_.getVPBit(mpc + 1, link + 1), aFD.qualityPacked(), 
				  aFD.wireGroupPacked(), aFD.stripPacked(),
				  aFD.patternPacked(), aFD.lrPacked(), myBX_);
    }
  return CSCCorrelatedLCTDigi();
}

int CSCTFTBFrontBlock::unpackData(unsigned short *buf,
				const CSCTFTBEventHeader& hdr)
{
//   cout << " CSCTFTBFrontBlock_unpackData-INFO: Unpacking event data" << endl;
   srdata_.clear();
   int nMPC = hdr.numMPC();
   int nLinks = hdr.numLinks();
   memcpy(&frontHeader_,buf,CSCTFTBFrontHeader::size()*sizeof(short));

   buf += CSCTFTBFrontHeader::size();
   size_ += CSCTFTBFrontHeader::size();


   for (int mpc = 1; mpc<=nMPC; mpc++) 
     {
       if((hdr.getActiveFront() & (1<<(mpc-1)))) 
	 {
	   std::vector<CSCTFTBFrontData> links_;
	   for(int link = 1; link<=nLinks;link++)
	     {
	       CSCTFTBFrontData data(mpc);
	       if(frontHeader_.getVPBit(mpc,link)||hdr.getZeroSupp()==0)
		 {		   
		   memcpy(&data,buf,CSCTFTBFrontData::size()*sizeof(short));
		   links_.push_back(data);
		   buf += CSCTFTBFrontData::size();
		   size_ += CSCTFTBFrontData::size();
		 }
	     }
	   srdata_.push_back(links_);
	 }
     }
   return 0;

}

/*
BitVector CSCTFTBFrontBlock::pack()
{
  BitVector result;

  BitVector header(reinterpret_cast<unsigned*>(&frontHeader_),frontHeader_.size()*sizeof(short)*8);
  result.assign(result.nBits(),header.nBits(),header);

  vector<vector<CSCTFTBFrontData> >::const_iterator fpgas;
  vector<CSCTFTBFrontData>::const_iterator links;

  for(fpgas = srdata_.begin();fpgas != srdata_.end();fpgas++)
    {
      for(links = fpgas->begin();links != fpgas->end();links++)
	{
	  BitVector lctdata_ = links->packVector();
	  result.assign(result.nBits(),lctdata_.nBits(),lctdata_);	  
	}
    }
  return result;
}
*/

std::ostream & operator<<(std::ostream & stream, const CSCTFTBFrontBlock & bx) 
{
  std::vector<std::vector<CSCTFTBFrontData> > sr = bx.frontData();
  stream << bx.frontHeader();
  if(bx.frontHeader().validPattern())
    stream << "\tFront Event Data (Track Stubs):\n";
  for(unsigned i = 1; i <= sr.size(); ++i)
    {
      if(sr[i-1].size()) stream<<"\tFPGA: "<<i<<'\n';
      for (unsigned j = 1;j <= sr[i-1].size();j++)
	{
	  stream<<"\t Link: "<<j<<' '<<std::dec
		<<" CSCID: " << bx.frontData(i,j).CSCIDPacked() << ' '
		<< bx.frontDigiData(i,j);
	}
    }
  return stream;
}


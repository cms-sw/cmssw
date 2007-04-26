// -*- C++ -*-
//
// Package:     CSCTFTBRawData
// Module:      CSCTFTBSPBlock
// 
// Description: SP Event Data Container class
//
// Implementation:
//     <Notes on implementation>
//
// Author:      Lindsey
// Created:     13.1.2005
//
// $Id: CSCTFTBSPBlock.cc,v 1.2 2006/06/22 14:46:05 lgray Exp $
//
// Revision History
// $Log: CSCTFTBSPBlock.cc,v $
// Revision 1.2  2006/06/22 14:46:05  lgray
// Forced commit of all code
//
// Revision 1.1  2006/06/22 00:34:18  lgray
// Moved all data format classes here. Removed old Packages from nightly
//
// Revision 1.1  2006/02/22 23:16:42  lgray
// First commit of test beam data format from UF
//
// Revision 1.5  2005/06/22 22:03:27  lgray
// update
//
// Revision 1.4  2005/05/13 08:29:59  lgray
// Another bug fix.
//
// Revision 1.3  2005/05/10 21:57:22  lgray
// Bugfixes, stability issues fixed
//
// Revision 1.2  2005/03/03 18:14:49  lgray
// Added ability to pack data back into raw form. Added test program for this as well.
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
#include <iostream>
#include <vector>
#include <string.h> //  memcpy

// Package include files
#include "EventFilter/CSCTFRawToDigi/src/CSCTFTBEventHeader.h"
#include "EventFilter/CSCTFRawToDigi/src/CSCTFTBSPBlock.h"

// External package include files
//#include "Utilities/GenUtil/interface/BitVector.h"

// STL classes

// Constants, enums and typedefs

// CVS-based strings (Id and Tag with which file was checked out)
static const char* const kIdString  = "$Id: CSCTFTBSPBlock.cc,v 1.2 2006/06/22 14:46:05 lgray Exp $";
static const char* const kTagString = "$Name:  $";

// Static data member definitions

// Constructors and destructor
CSCTFTBSPBlock::CSCTFTBSPBlock()
{
  myBX_ = 0;
  size_ = 0;
}

//
CSCTFTBSPBlock::CSCTFTBSPBlock(unsigned short *buf, int bx, 
				 const CSCTFTBEventHeader& hdr)
{
  myBX_ = bx;
  size_ = 0;
  unpackData(buf, hdr);
}

CSCTFTBSPBlock::CSCTFTBSPBlock(const CSCTFTBSPBlock &parent)
{   
  //  spdata_ = parent.spdata_;
  /*for(int i = 0 ; i< parent.spdata_.size(); i++)
    {
      CSCTFTBSPData temp = parent.spdata_[i];
      spdata_.push_back(temp);
      }*/
  spdata_ = parent.spdata_;
  dtdata_ = parent.dtdata_;
  spheaderdata_ = parent.spheaderdata_;
  myBX_ = parent.myBX_;
  size_ = parent.size_;
  
}

CSCTFTBSPBlock::~CSCTFTBSPBlock()
{
  spdata_.clear();
  dtdata_.clear();
}

// Member Functions

int CSCTFTBSPBlock::unpackData(unsigned short *buf, 
				const CSCTFTBEventHeader& thehdr)
{
  bool pt_present = false;
  unsigned mode = 0,link = 0;
  // store header data  
  memcpy(&spheaderdata_,buf,CSCTFTBSPHeader::size()*sizeof(short));
  buf += CSCTFTBSPHeader::size();
  size_ += CSCTFTBSPHeader::size();
  
  //store DT data and SP data....
  
  if(thehdr.getActiveDT()!=0) {
    for(unsigned i=1;i<=2;i++)
      {
	if(spheaderdata_.getVPDTBit(i)||thehdr.getZeroSupp()==0)
	  {
	    CSCTFTBDTData dtdata;
	    memcpy(&dtdata,buf,CSCTFTBDTData::size()*sizeof(short));
	    dtdata_.push_back(dtdata);
	    buf += CSCTFTBDTData::size();
	    size_ += CSCTFTBDTData::size();
	  }
      }       
  }
     
  for(unsigned i=1;i<=3;i++)
    {
      if(spheaderdata_.getTrackMode(i)>0 || thehdr.getZeroSupp() == 0)
	{
	  mode = spheaderdata_.getTrackMode(i);
	  link = i;
	  pt_present = (i == thehdr.getPtLutSpy());
	  CSCTFTBSPData spdata(buf,pt_present,mode,link);  
	  spdata_.push_back(spdata);
	  buf += (pt_present) ? CSCTFTBSPData::size()+CSCTFTBPTData::size() : 
	    CSCTFTBSPData::size();
	  size_ += (pt_present) ? CSCTFTBSPData::size()+CSCTFTBPTData::size() :
	    CSCTFTBSPData::size();
	}
    }

   return 0;

}

/*
BitVector CSCTFTBSPBlock::pack()
{
  BitVector result;
  
  BitVector header(reinterpret_cast<const unsigned*>(&spheaderdata_),
		   spheaderdata_.size()*sizeof(short)*8);
  result.assign(result.nBits(),header.nBits(),header);

  vector<CSCTFTBDTData>::const_iterator DTiter;
  vector<CSCTFTBSPData>::const_iterator SPiter;

  for(DTiter = dtdata_.begin();DTiter != dtdata_.end();DTiter++)
    {
      BitVector dt = DTiter->packVector();
      result.assign(result.nBits(),dt.nBits(),dt);
    }

  for(SPiter = spdata_.begin();SPiter != spdata_.end();SPiter++)
    {
      BitVector sp = SPiter->packVector();
      result.assign(result.nBits(),sp.nBits(),sp);
    }

  return result;
}
*/

std::ostream & operator<<(std::ostream & stream, const CSCTFTBSPBlock & bx) 
{ 
  //vector<CSCTFTBSPData> sp = bx.spData();
  std::vector<CSCTFTBDTData> dt = bx.dtData();
  CSCTFTBSPHeader hdr = bx.spHeader();
  stream << hdr;
  if(hdr.getTrackMode(1)||hdr.getTrackMode(2)||hdr.getTrackMode(3)
     ||hdr.getVPDTBit(1)||hdr.getVPDTBit(2))
    stream<<"\tSP Event Data (Tracks and DT Track Stubs):\n";
  for(unsigned i=0;i<dt.size();i++)
    {
      stream<<"\t Link "<< i << dt[i];
    }
  for(unsigned j=0;j<bx.spData().size();j++)
    {
      stream<<bx.spData()[j];
    }
  return stream;
}


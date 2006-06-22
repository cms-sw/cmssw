// -*- C++ -*-
// //
// Package:     CSCTFTBRawFormat
// Module:      CSCTFTBSPData
// 
// Description: SP Event Data class
//
// Implementation:
//     <Notes on implementation>
//
// Author:      Lindsey Gray
// Created:     13.1.2005
//
// $Id: CSCTFTBSPData.cc,v 1.1 2006/06/22 00:34:18 lgray Exp $
//
// Revision History
// $Log: CSCTFTBSPData.cc,v $
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
// Revision 1.21  2004/10/12 00:09:07  tfcvs
// DEA: update to Ntuple
//
// Revision 1.20  2004/06/13 12:06:14  tfcvs
// DEA: closer to having beam dat ain sim
//
// Revision 1.19  2004/06/09 10:04:52  tfcvs
// DEA
//
// Revision 1.18  2004/06/07 22:37:23  tfcvs
// DEA
//
// Revision 1.17  2004/05/21 10:17:27  tfcvs
// DEA: changes to analysis
//
// Revision 1.16  2004/05/18 22:25:54  tfcvs
// DEA: change in location of Pt LUT
//
// Revision 1.15  2004/05/18 21:53:42  tfcvs
// DEA: some print out
//
// Revision 1.14  2004/05/18 15:00:25  tfcvs
// DEA: close to new SP data format
//
// Revision 1.13  2004/05/18 08:00:34  tfcvs
// DEA: touch base
//
// 
//

// System include files
#include <string.h> // memcpy
#include <iostream>
// Package include files
#include "EventFilter/CSCTFRawToDigi/src/CSCTFTBSPData.h"

// External package include files

// STL classes

// Constants, enums and typedefs

// CVS-based strings (Id and Tag with which file was checked out)
static const char* const kIdString  = "$Id: CSCTFTBSPData.cc,v 1.1 2006/06/22 00:34:18 lgray Exp $";
static const char* const kTagString = "$Name:  $";

// Static data member definitions

// Constructors and destructor
CSCTFTBSPData::CSCTFTBSPData():
  zero1_(0),
  zero2_(0),
  zero3_(0),
  pt_data_(NULL),
  mymode_(0),
  mylink_(0)
{}

CSCTFTBSPData::~CSCTFTBSPData()
{  
  if(pt_data_)
    {
      delete pt_data_;
      pt_data_ = NULL;
    }  
}

CSCTFTBSPData::CSCTFTBSPData(const CSCTFTBSPData & parent)
{
  this->pt_data_ = NULL;
  if(&parent)
    {      
      memcpy(this,&parent,sizeof(CSCTFTBSPData));   
    }
  if(parent.pt_data_)
    { 
      pt_data_ = new CSCTFTBPTData();
      memcpy(pt_data_,parent.pt_data_,CSCTFTBPTData::size()*sizeof(short)); 
    }
}

CSCTFTBSPData::CSCTFTBSPData(unsigned short * buf, bool ptflag,
			     unsigned themode_,unsigned link_)
{
  pt_data_ = NULL;
  unpackData(buf, ptflag);  
  mymode_ = themode_;
  mylink_ = link_;
}

CSCTFTBSPData CSCTFTBSPData::operator=(const CSCTFTBSPData & parent)
{  
  if(this == &parent) return *this;
  memcpy(this,&parent,sizeof(CSCTFTBSPData));
  if(parent.pt_data_)
    {      
      memcpy(pt_data_,parent.pt_data_,CSCTFTBPTData::size()*sizeof(short));
    }
  return *this;
}

int CSCTFTBSPData::unpackData(unsigned short *buf,bool ptflag)
{  
  memcpy(this,buf,size()*sizeof(short));
  if (ptflag) 
    {      
      pt_data_ = new CSCTFTBPTData();
      memcpy(pt_data_,buf+size(),CSCTFTBPTData::size()*sizeof(short));
    }
  return 0;
}

/*
BitVector CSCTFTBSPData::packVector() const
{
  BitVector result;
  BitVector spdata(reinterpret_cast<const unsigned*>(this),size()*sizeof(short)*8);
  result.assign(result.nBits(),spdata.nBits(),spdata);

  if(pt_data_)
    {
      BitVector pt(reinterpret_cast<const unsigned*>(pt_data_),size()*sizeof(short)*8);
      result.assign(result.nBits(),pt.nBits(),pt);
    }
  return result;
}
*/

std::ostream & operator<<(std::ostream & stream, const CSCTFTBSPData & spData) 
{
  stream << "\t Track Data:\n";
  if(spData.pt_data_)
    {
      stream <<*(spData.pt_data_);
    }
  return stream;
}

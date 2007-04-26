// -*- C++ -*-
// //
// Package:     CSCTFTBRawFormat
// Module:      CSCTFTBPTData
// 
// Description: PT Data class
//
// Implementation:
//     <Notes on implementation>
//
// Author:      Lindsey Gray
// Created:     13.1.2005
//
// $Id: CSCTFTBPTData.cc,v 1.2 2006/06/22 14:46:05 lgray Exp $
//
// Revision History
// $Log: CSCTFTBPTData.cc,v $
// Revision 1.2  2006/06/22 14:46:05  lgray
// Forced commit of all code
//
// Revision 1.1  2006/06/22 00:34:18  lgray
// Moved all data format classes here. Removed old Packages from nightly
//
// Revision 1.1  2006/02/22 23:16:42  lgray
// First commit of test beam data format from UF
//
// Revision 1.2  2005/05/10 21:57:22  lgray
// Bugfixes, stability issues fixed
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
#include <string.h> // memset
// Package include files
#include "EventFilter/CSCTFRawToDigi/src/CSCTFTBPTData.h"

// External package include files

// STL classes

// Constants, enums and typedefs

// CVS-based strings (Id and Tag with which file was checked out)
static const char* const kIdString  = "$Id: CSCTFTBPTData.cc,v 1.2 2006/06/22 14:46:05 lgray Exp $";
static const char* const kTagString = "$Name:  $";

// Static data member definitions

// Constructors and destructor
CSCTFTBPTData::CSCTFTBPTData()
{}

/*CSCTFTBPTData::~CSCTFTBPTData()
{}*/

CSCTFTBPTData::CSCTFTBPTData(const CSCTFTBPTData & parent)
{
  //rear_pt_ = parent.rear_pt_;
  //front_pt_ = parent.front_pt_;
  memcpy(this,&parent,size()*sizeof(short));

}

CSCTFTBPTData CSCTFTBPTData::operator=(const CSCTFTBPTData & parent)
{
  memcpy(this,&parent,size()*sizeof(short));
  //rear_pt_ = parent.rear_pt_;
  //front_pt_ = parent.front_pt_;
  return *this;
}

unsigned int CSCTFTBPTData::ptLUT(int fr ) const 
{
  fr = (fr == 0 || fr ==1) ? fr : 0;
  // Rear
  if (fr == 0) return rear_pt_;
  // Front
  else         return front_pt_;

}

std::ostream & operator<<(std::ostream & stream, const CSCTFTBPTData & ptData) 
{
  stream << "\t\tFront PT data : " << std::hex << ptData.ptLUT(1) << std::dec << std::endl;
  stream << "\t\tRear PT data  : " << std::hex << ptData.ptLUT(0) << std::dec << std::endl;
  return stream;
}

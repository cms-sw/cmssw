// -*- C++ -*-
// //
// Package:     CSCTFTBRawFormat
// Module:      CSCTFTBFrontData
// 
// Description: Front Event Data class
//
// Implementation:
//     <Notes on implementation>
//
// Author:      Lindsey Gray
// Created:     13.1.2005
//
// $Id: CSCTFTBFrontData.cc,v 1.1 2006/06/22 00:34:18 lgray Exp $
//
// Revision History
// $Log: CSCTFTBFrontData.cc,v $
// Revision 1.1  2006/06/22 00:34:18  lgray
// Moved all data format classes here. Removed old Packages from nightly
//
// Revision 1.2  2006/02/27 10:43:57  lgray
// Print changes.
//
// Revision 1.1  2006/02/22 23:16:42  lgray
// First commit of test beam data format from UF
//
// Revision 1.2  2005/03/03 18:14:48  lgray
// Added ability to pack data back into raw form. Added test program for this as well.
//
// Revision 1.1  2005/02/14 21:01:32  lgray
// First Commit from UF
//
// Revision 1.16  2004/05/28 00:24:58  tfcvs
// DEA: a working version of code for 4 chambers!
//
// Revision 1.15  2004/05/21 10:17:27  tfcvs
// DEA: changes to analysis
//
// Revision 1.14  2004/05/19 17:27:32  tfcvs
// DEA: touch base with LCT reformat
//
// Revision 1.13  2004/05/18 15:00:25  tfcvs
// DEA: close to new SP data format
//
// Revision 1.12  2004/05/18 11:37:46  tfcvs
// DEA: touch base
//
// Revision 1.11  2004/05/18 08:00:35  tfcvs
// DEA: touch base
//
// Revision 1.10  2004/05/16 07:43:49  tfcvs
// DEA: TB2003 version working with new software
//
// Revision 1.9  2003/09/19 20:22:55  tfcvs
// latest
//
// Revision 1.8  2003/08/27 22:09:19  tfcvs
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

// Package include files
#include "EventFilter/CSCTFRawToDigi/src/CSCTFTBFrontData.h"

// External package include files
//#include "Utilities/GenUtil/interface/BitVector.h"

// STL classes

// Constants, enums and typedefs

// CVS-based strings (Id and Tag with which file was checked out)
static const char* const kIdString  = "$Id: CSCTFTBFrontData.cc,v 1.1 2006/06/22 00:34:18 lgray Exp $";
static const char* const kTagString = "$Name:  $";

// Static data member definitions

// Constructors and destructor
CSCTFTBFrontData::CSCTFTBFrontData(unsigned mpc): myMPC_(mpc)
{
  memset(this,0,size()*sizeof(short));
}

CSCTFTBFrontData::~CSCTFTBFrontData()
{
  memset(this,0,size()*sizeof(short));
  myMPC_ = 0;
}

CSCTFTBFrontData::CSCTFTBFrontData(const CSCTFTBFrontData& parent)
{
  memcpy(this,&parent,size()*sizeof(unsigned short));
  myMPC_ = parent.getMPC();
}

/*
BitVector CSCTFTBFrontData::packVector() const
{
  return BitVector(reinterpret_cast<const unsigned*>(this),size()*sizeof(short)*8);
}
*/

std::ostream & operator<<(std::ostream & stream, const CSCTFTBFrontData & srData) {
  stream << "Track Stub:\n\t  " <<"NEED PRETTY PRINT\n";//srData.corrLCTData();  
  return stream;
}

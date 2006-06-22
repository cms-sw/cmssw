#ifndef CSCTFTBRAWFORMAT_L1MUCSCEVENTDATA_H
#define CSCTFTBRAWFORMAT_L1MUCSCEVENTDATA_H
// -*- C++ -*-
//
// Package:     CSCTFTBTBRawFormat
// Module:      CSCTFTBEventData
// 
// Description: Header file for SP VME Event Data class
//
// Implementation:
//     <Notes on implementation>
//
// Author:      Lindsey Gray
// Created:     13.1.2005
//
// $Id: CSCTFTBEventData.h,v 1.1 2006/02/22 23:15:49 lgray Exp $
//
// Revision History
// $Log: CSCTFTBEventData.h,v $
// Revision 1.1  2006/02/22 23:15:49  lgray
// First commit of test beam data format from UF
//
// Revision 1.3  2005/03/03 18:14:48  lgray
// Added ability to pack data back into raw form. Added test program for this as well.
//
// Revision 1.2  2005/02/15 19:13:08  lgray
// Getting Ready for DDU, working on new root tree maker
//
// Revision 1.1  2005/02/14 20:59:46  lgray
// First Commit from UF
//
// Revision 1.4  2004/11/29 23:01:03  tfcvs
// (LAG) Added static bool to account for data with CDF header/trailer
//
// Revision 1.3  2004/05/18 08:00:10  tfcvs
// DEA: touch base
//
// Revision 1.2  2004/05/17 08:25:52  tfcvs
// DEA: switch to SR BX data
//
// Revision 1.1  2004/05/16 12:17:17  tfcvs
// DEA: call overall data block SRSP
//
// Revision 1.10  2003/08/27 22:07:56  tfcvs
// Added pretty-print - Rick
//
// Revision 1.9  2003/05/27 12:45:00  tfcvs
// add BXN and L1A # to SPEvent -DEA
//
// Revision 1.8  2003/05/25 10:13:02  tfcvs
// first working version -DEA
//
// Revision 1.7  2003/05/20 22:13:06  tfcvs
// HS - Added Darin's changes
//
// Revision 1.6  2003/05/19 23:23:12  tfcvs
// HS - Commit after some changes
//
// Revision 1.4  2003/05/19 15:47:18  tfcvs
// HS - Some cleanup
//
// Revision 1.3  2003/05/19 00:25:56  tfcvs
// DEA: committed, but may not compile
//
// Revision 1.2  2003/05/15 23:58:40  tfcvs
// HS - Some cosmetics
//
//
//

// System include files

// Package include files
#include "EventFilter/CSCTFRawToDigi/src/CSCTFTBEventHeader.h"
#include "EventFilter/CSCTFRawToDigi/src/CSCTFTBFrontBlock.h"
#include "EventFilter/CSCTFRawToDigi/src/CSCTFTBSPBlock.h"

// External package include files

// STL classes
#include <iostream>
#include <vector>

// Forward declarations


class CSCTFTBEventData 
{

// Friend classses and functions

// Public part
   public:
        // Constants, enums and typedefs

        // Constructors and destructor
        CSCTFTBEventData();
        explicit CSCTFTBEventData(unsigned short *buf);
	//CSCTFTBEventData(const CSCTFTBEventData&);	
	
	~CSCTFTBEventData();
	
	//CSCTFTBEventData operator=(const CSCTFTBEventData&);

        // Member functions
  
        // Const member functions
	
	unsigned getTriggerBX() const {return trigBX_;}
	unsigned size() const {return size_;}
	void setEventInformation(unsigned,unsigned);
	/// Sector Receiver and Sector Processor data
        std::vector<CSCTFTBFrontBlock> frontData() const {return frontBlocks_;}
	CSCTFTBFrontBlock frontDatum(unsigned bx = trigBX_) const 
	  {return frontBlocks_[bx = (bx > 0 && bx<= frontBlocks_.size()) ? bx-1 : 0];}

	std::vector<CSCTFTBSPBlock> spData() const {return spBlocks_;}
	CSCTFTBSPBlock spDatum(unsigned bx = trigBX_) const 
	  {return spBlocks_[bx = (bx > 0 && bx<= spBlocks_.size()) ? bx-1 : 0];}
	
	/// This SP's event header
	CSCTFTBEventHeader eventHeader() const {return spHead_;}

        // Static member functions
        enum {SAMPLESIZE = 16};
	static void setTriggerBX(unsigned theBX) {trigBX_ = theBX;}
	static void onlyTriggerBX(bool disp) {onlyTrigBX_ = disp;}
	static void setCDF(bool hasCDF) {hasCDF_ = hasCDF;}

	//returns the binary event data
	//BitVector pack();

        /// Trailer info
        unsigned short *SP_tail;
        int             SPbytes,errorstat2; 
  
        friend std::ostream & operator<<(std::ostream & stream, const CSCTFTBEventData &);
// Private part
   private:

        // Protected member functions
        int unpackData(unsigned short * buf);

        // Protected const member functions

        /// Front and SP Data is unpacked and stored in these vectors
	std::vector<CSCTFTBFrontBlock> frontBlocks_;
	std::vector<CSCTFTBSPBlock> spBlocks_;
	/// This SP's event header is stored here
	CSCTFTBEventHeader spHead_;
	
        // Constructors and destructor
	
        // Assignment operator(s)

        // Private member functions

        // Private const member functions

        // Data members
	unsigned size_;
        // Static data members
	static bool hasCDF_; 
	static bool onlyTrigBX_;
	static unsigned trigBX_;
        // Inline function definitions

};

#endif

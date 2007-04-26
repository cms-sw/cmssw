#if !defined(CSCTFTBRAWFORMAT_CSCTFTBSPBLOCK_H)
#define CSCTFTBRAWFORMAT_CSCTFTBSPBLOCK_H
// -*- C++ -*-
//
// Package:     CSCTFTBRawFormat
// Module:      CSCTFTBSPBlock
// 
// Description: Header file for SP Data Block (Tracks and DT Stubs)
//
// Implementation:
//     <Notes on implementation>
//
// Author:      Lindsey Gray
// Created:     13.1.2005
//
// $Id: CSCTFTBSPBlock.h,v 1.3 2006/06/22 14:51:39 lgray Exp $
//
// Revision History
// $Log: CSCTFTBSPBlock.h,v $
// Revision 1.3  2006/06/22 14:51:39  lgray
// Found error.
//
// Revision 1.2  2006/06/22 14:46:05  lgray
// Forced commit of all code
//
// Revision 1.1  2006/06/22 00:34:18  lgray
// Moved all data format classes here. Removed old Packages from nightly
//
// Revision 1.1  2006/02/22 23:15:49  lgray
// First commit of test beam data format from UF
//
// Revision 1.3  2005/05/10 21:57:21  lgray
// Bugfixes, stability issues fixed
//
// Revision 1.2  2005/03/03 18:14:48  lgray
// Added ability to pack data back into raw form. Added test program for this as well.
//
// Revision 1.1  2005/02/14 20:59:46  lgray
// First Commit from UF
//
// Revision 1.6  2004/05/18 15:00:25  tfcvs
// DEA: close to new SP data format
//
// Revision 1.5  2004/05/18 11:37:46  tfcvs
// DEA: touch base
//
// Revision 1.4  2004/05/18 09:45:34  tfcvs
// DEA: touch base
//
// Revision 1.3  2004/05/18 08:00:10  tfcvs
// DEA: touch base
//
// Revision 1.2  2003/08/27 22:07:51  tfcvs
// Added pretty-print - Rick
//
// Revision 1.1  2003/05/25 10:13:02  tfcvs
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
#include <vector>
#include <iostream>

// Package include files
//class BitVector;
#include "EventFilter/CSCTFRawToDigi/src/CSCTFTBSPHeader.h"
#include "EventFilter/CSCTFRawToDigi/src/CSCTFTBSPData.h"
#include "EventFilter/CSCTFRawToDigi/src/CSCTFTBDTData.h"

// External package include files

// STL classes

// Forward declarations
class CSCTFTBEventHeader;

class CSCTFTBSPBlock 
{

// Friend classses and functions

// Public part
   public:
        // Constants, enums and typedefs

        // Constructors and destructor
        CSCTFTBSPBlock();

        CSCTFTBSPBlock(unsigned short *buf, int bx,
				const CSCTFTBEventHeader& hdr); 

	CSCTFTBSPBlock(const CSCTFTBSPBlock&);
	~CSCTFTBSPBlock();

        // Member functions
  
        // Const member functions
	unsigned size() const {return size_;}
	/// return the SP data 
        std::vector<CSCTFTBSPData> spData() const {return spdata_;}
	CSCTFTBSPData spData(unsigned link) const
	  {return spdata_[link = (link > 0 && link<= spdata_.size()) ? link-1 : 0];}

	/// return the DT data 
        std::vector<CSCTFTBDTData> dtData() const {return dtdata_;}
	CSCTFTBDTData dtData(unsigned link) const
	  {return dtdata_[link = (link > 0 && link<= dtdata_.size()) ? link-1 : 0];}

	/// return the SP Header data
	CSCTFTBSPHeader spHeader() const {return spheaderdata_;}

	/// (relative) BX assigned to this bank
	int BX() const {return myBX_;};

	/// make a bit vector
	//BitVector pack();

        /// pretty-print
        friend std::ostream & operator<<(std::ostream & stream, const CSCTFTBSPBlock &);
// Protected part
   protected:

        // Protected member functions
        int unpackData(unsigned short * buf, 
		       const CSCTFTBEventHeader& the_hdr);

        // Protected const member functions

        /// SP Data is unpacked and stored in this vector
	std::vector<CSCTFTBSPData> spdata_;

        /// DT Data is unpacked and stored in this vector
	std::vector<CSCTFTBDTData> dtdata_;
	
	/// SP Header Data is unpacked and stored in this
	CSCTFTBSPHeader spheaderdata_;

	int myBX_;
	unsigned size_;
// Private part
   private:

        // Constructors and destructor

        // Assignment operator(s)

        // Private member functions

        // Private const member functions

        // Data members

        // Static data members

        // Inline function definitions

};

#endif

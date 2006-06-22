#if !defined(CSCTFTBRAWFORMAT_CSCTFTBSPHEADER_H)
#define CSCTFTBRAWFORMAT_CSCTFTBSPHEADER_H
// -*- C++ -*-
//
// Package:     CSCTFTBRawData
// Module:      CSCTFTBSPHeader
// 
// Description: Header file for SP Data Block Header
//
// Implementation:
//     <Notes on implementation>
//
// Author:      Lindsey Gray
// Created:     13.1.2005
//
// $Id: CSCTFTBSPHeader.h,v 1.1 2006/02/22 23:15:49 lgray Exp $
//
// Revision History
// $Log: CSCTFTBSPHeader.h,v $
// Revision 1.1  2006/02/22 23:15:49  lgray
// First commit of test beam data format from UF
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
#include <iostream>
#include <string.h> //memset
// Package include files

// External package include files

// STL classes

// Forward declarations

class CSCTFTBSPHeader 
{

// Friend classses and functions

// Public part
   public:
        // Constants, enums and typedefs

        // Constructors and destructor
        CSCTFTBSPHeader();

        CSCTFTBSPHeader(const CSCTFTBSPHeader&);

	~CSCTFTBSPHeader();
	
	CSCTFTBSPHeader operator=(const CSCTFTBSPHeader&);
        // Member functions
  
        // Const member functions

	/// simple data integrity check
	bool check() const {return ((zero1_ == 0) && (zero2_ == 0));}

	/// size of data bank in 16-bit words
        static unsigned size() {return size_;};

	/// Synch Error word (15 links/bits)
	unsigned int synchError() const {return  synch_err_;}

	/// Synch Error for given link
	unsigned int getSEBit(unsigned int frontFPGA, unsigned int link) const;

	/// Track Mode (i.e. quality)  for given track
	unsigned int getTrackMode(unsigned int trk) const;

	/// Valid Pattern for given DT link
	unsigned int getVPDTBit(unsigned int link) const;

        /// pretty-print
        friend std::ostream & operator<<(std::ostream & stream, const CSCTFTBSPHeader &);
// Protected part
   protected:

        // Protected member functions

        // Protected const member functions

// Private part
   private:

	//frame 1
	unsigned synch_err_  :15;
	unsigned zero1_      :1; //always zero
	//frame 2
	unsigned mode1_      :4;
	unsigned mode2_      :4;
	unsigned mode3_      :4;
	unsigned MB1A_flag_  :1;
	unsigned MB1D_flag_  :1;
	unsigned zero2_      :2; //always zero

        // Constructors and destructor

        // Assignment operator(s)

        // Private member functions

        // Private const member functions

        // Data members

        // Static data members
	static const unsigned size_ = 2;
        // Inline function definitions

};

#endif

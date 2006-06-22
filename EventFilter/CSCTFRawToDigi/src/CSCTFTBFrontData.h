#if !defined(CSCTFTBRAWFORMAT_CSCTFTBFRONTDATA_H)
#define CSCTFTBRAWFORMAT_CSCTFTBFRONTDATA_H
// -*- C++ -*-
//
// Package:     CSCTFTBRawFormat
// Module:      CSCTFTBFrontData
// 
// Description: Header file for Front Event Data (Correlated LCTs)
//
// Implementation:
//     <Notes on implementation>
//
// Author:      Lindsey Gray
// Created:     13.1.2005
//
// $Id: CSCTFTBFrontData.h,v 1.1 2006/06/22 00:34:18 lgray Exp $
//
// Revision History
// $Log: CSCTFTBFrontData.h,v $
// Revision 1.1  2006/06/22 00:34:18  lgray
// Moved all data format classes here. Removed old Packages from nightly
//
// Revision 1.2  2006/02/24 23:15:43  lgray
// Fixed buildfile, added default ctor to FrontData
//
// Revision 1.1  2006/02/22 23:15:49  lgray
// First commit of test beam data format from UF
//
// Revision 1.2  2005/03/03 18:14:48  lgray
// Added ability to pack data back into raw form. Added test program for this as well.
//
// Revision 1.1  2005/02/14 20:59:46  lgray
// First Commit from UF
//
// Revision 1.13  2004/05/21 10:17:26  tfcvs
// DEA: changes to analysis
//
// Revision 1.12  2004/05/19 17:27:32  tfcvs
// DEA: touch base with LCT reformat
//
// Revision 1.11  2004/05/18 11:37:46  tfcvs
// DEA: touch base
//
// Revision 1.10  2004/05/18 08:00:10  tfcvs
// DEA: touch base
//
// Revision 1.9  2003/08/27 22:08:13  tfcvs
// Added pretty-print - Rick
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
#include <iostream>
// Package include files

// External package include files

// STL classes

// Forward declarations
//class BitVector;

class CSCTFTBFrontData 
{

// Friend classses and functions

// Public part
   public:
        // Constants, enums and typedefs

        // Constructors and destructor
        CSCTFTBFrontData() { }
        CSCTFTBFrontData(unsigned);
        CSCTFTBFrontData(const CSCTFTBFrontData&);

	~CSCTFTBFrontData();

        // Member functions
	
        // Const member functions
	
	/// return quality bits
	unsigned qualityPacked() const {return quality_;}
	
	/// return CSC ID bits
	unsigned CSCIDPacked() const {return csc_id_;}

	/// return Wire Group bits
	unsigned wireGroupPacked() const {return wire_group_;}

	/// return clct pattern bits
	unsigned stripPacked() const {return clct_patrn_;}

	/// return clct pattern number bits
	unsigned patternPacked() const {return clct_patnum_;}

	/// return l/r bit
	unsigned lrPacked() const {return l_r_;}

	/// return BX0 bit
	unsigned bx0Packed() const {return BX0_;}

	/// return BC0 bit
	unsigned bc0Packed() const {return BC0_;}

	/// return MPC attached to this SR
	int getMPC() const { return myMPC_;}

	/// Simple data integrity check
	bool check() {return (zero1_ == 0 && zero2_ == 0);}

        /// pretty-print
        friend std::ostream & operator<<(std::ostream & stream, const CSCTFTBFrontData &);
	/// put into BitVector
	//BitVector packVector() const;

        // Static member functions
	static int size() {return size_;};

// Private part
   private:

	//frame 1
	unsigned quality_    :4;
	unsigned csc_id_     :4;
	unsigned wire_group_ :7;
	unsigned zero1_      :1; // always zero
	//frame 2
	unsigned clct_patrn_ :8;
	unsigned clct_patnum_:4;
	unsigned l_r_        :1;
	unsigned BX0_        :1;
	unsigned BC0_        :1;
	unsigned zero2_      :1; //always zero

        // Constructors and destructor

        // Assignment operator(s)

        // Private member functions

        // Private const member functions

        // Data members
	int myMPC_;
        // Static data members
	static const unsigned size_ = 2; // 16 bit words
        // Inline function definitions

};

#endif

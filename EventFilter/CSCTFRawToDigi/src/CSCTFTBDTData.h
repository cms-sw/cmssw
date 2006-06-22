#if !defined(CSCTFTBRAWFORMAT_L1MUCSCDTDATA_H)
#define CSCTFTBRAWFORMAT_L1MUCSCDTDATA_H
// -*- C++ -*-
//
// Package:     CSCTFTBRawData
// Module:      CSCTFTBDTData
// 
// Description: Header file for DT Event Data sent to SP,
//              rewritten to use bitfields.
// Implementation:
//     <Notes on implementation>
//
// Author:      Lindsey Gray
// Created:     13.1.2005
//
// $Id: CSCTFTBDTData.h,v 1.1 2006/06/22 00:34:18 lgray Exp $
//
// Revision History
// $Log: CSCTFTBDTData.h,v $
// Revision 1.1  2006/06/22 00:34:18  lgray
// Moved all data format classes here. Removed old Packages from nightly
//
// Revision 1.2  2006/02/27 01:19:23  lgray
// Changes to print functions and test data file
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
// Revision 1.3  2004/11/04 21:24:37  tfcvs
// DEA: bug fix
//
// Revision 1.2  2004/05/21 10:17:26  tfcvs
// DEA: changes to analysis
//
// Revision 1.1  2004/05/18 11:37:46  tfcvs
// DEA: touch base
//
//

// System include files
#include <iostream>

// Package include files

// External package include files

class BitVector;
// STL classes

// Forward declarations

class CSCTFTBDTData 
{

// Friend classses and functions

// Public part
   public:
        // Constants, enums and typedefs

        // Constructors and destructor
        CSCTFTBDTData();
	CSCTFTBDTData(unsigned quality, unsigned bend, unsigned flag,
		      unsigned calib, unsigned phi, unsigned bx, unsigned bc0);
	CSCTFTBDTData(const CSCTFTBDTData&);
	
	~CSCTFTBDTData();

        // Member functions

        // Const member functions	  
	/// check to see if data format is ok (1 = good, 0 = bad)
	bool check() const {return !(zero1_||zero2_||zero3_||zero4_);} 
	
	unsigned qualityPacked() const {return quality_;}
	unsigned phiBendPacked() const {return bend_;}   
	unsigned phiPacked()     const {return phi_;}    
	unsigned bxPacked()      const {return bx_;}     
	unsigned bc0Packed()     const {return bc0_;}     
	
        /// pretty-print
        friend std::ostream & operator<<(std::ostream & stream, const CSCTFTBDTData &);

	/// make a bitvector
        //BitVector packVector() const;

	// Static member functions
	static unsigned size() {return size_;}

// Private part
   private:
	//Bit Fields
	//frame 1
	unsigned quality_    :3;
	unsigned zero1_      :1; //always zero
	unsigned bend_       :5;
	unsigned zero2_      :3; //always zero
	unsigned flag_       :1;
	unsigned calib_      :1;
	unsigned zero3_      :2; //always zero
	//frame 2
	unsigned phi_        :12;
	unsigned bx_         :2;
	unsigned bc0_        :1;
	unsigned zero4_      :1; //always zero

        // Constructors and destructor

        // Assignment operator(s)

        // Private member functions

        // Private const member functions

        // Data members	
	        
	// Static data members
	static const unsigned size_ = 2; // 16 bit words
        
	// Inline function definitions
};

#endif

#if !defined(CSCTFTBRAWFORMAT_CSCTFTBSPDATA_H)
#define CSCTFTBRAWFORMAT_CSCTFTBSPDATA_H
// -*- C++ -*-
//
// Package:     CSCTFTBRawFormat
// Module:      CSCTFTBSPData
// 
// Description: Header file for SP Event Data (Tracks)
//
// Implementation:
//     <Notes on implementation>
//
// Author:      Lindsey Gray
// Created:     13.1.2005
//
// $Id: CSCTFTBSPData.h,v 1.1 2006/06/22 00:34:18 lgray Exp $
//
// Revision History
// $Log: CSCTFTBSPData.h,v $
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
// Revision 1.13  2004/05/21 10:17:26  tfcvs
// DEA: changes to analysis
//
// Revision 1.12  2004/05/18 15:00:25  tfcvs
// DEA: close to new SP data format
//
// Revision 1.11  2004/05/18 08:00:10  tfcvs
// DEA: touch base
//

// System include files
#include <iostream>

// Package include files
#include "EventFilter/CSCTFRawToDigi/src/CSCTFTBPTData.h"

// External package include files

// STL classes

// Forward declarations
//class BitVector;

class CSCTFTBSPData 
{
// Friend classses and functions

// Public part
   public:
        // Constants, enums and typedefs

        // Constructors and destructor
        CSCTFTBSPData();
        CSCTFTBSPData(unsigned short * buf,
			       bool ptflag,
			       unsigned themode_,
			       unsigned link_);	
	CSCTFTBSPData(const CSCTFTBSPData &);
	~CSCTFTBSPData();

	CSCTFTBSPData operator=(const CSCTFTBSPData &);

        // Member functions
        static unsigned size() {return size_;};

        // Const member functions

	/// Specific bit pattern of track 
        unsigned int trackId() const {return trkId_;}

	/// Return link number
	unsigned int link() const {return mylink_;}

	/// Return PT data, wrapped in a class
	CSCTFTBPTData* ptData() const {return pt_data_;}
	
	///accessors to bits	
	unsigned phiPacked()    const {return phi_;}
	unsigned frPacked()     const {return fr_;}
	unsigned chargePacked() const {return chrg_;}
	unsigned haloPacked()   const {return hl_;}
	unsigned etaPacked()    const {return eta_;}
	unsigned sePacked()     const {return se_;}
	unsigned rsvPacked()    const {return rsv_;}
	unsigned dphi12Packed() const {return dphi_12_;}
	unsigned dphi23Packed() const {return dphi_23_;}
	unsigned signPacked()   const {return sign_;}
	unsigned bx0Packed()    const {return bx0_;}
	unsigned bc0Packed()    const {return bc0_;}

	///put into bit vector
	//BitVector packVector() const;

        /// pretty-print
        friend std::ostream & operator<<(std::ostream & stream, const CSCTFTBSPData &);

        // Static member functions
		       
// Private part

private:

	//frame 1
	unsigned phi_        :5;
	unsigned fr_         :1;
	unsigned chrg_       :1;
	unsigned hl_         :1;
	unsigned eta_        :5;
	unsigned se_         :1;
	unsigned rsv_        :1;
	unsigned zero1_      :1; //always zero
	//frame 2
	unsigned dphi_12_    :8;
	unsigned dphi_23_    :4;
	unsigned sign_       :1;
	unsigned bx0_        :1;
	unsigned bc0_        :1;
	unsigned zero2_      :1; //always zero
	//frame 3
	unsigned trkId_      :15;
	unsigned zero3_      :1; //always zero
	
	CSCTFTBPTData *pt_data_;
	unsigned mymode_;
	unsigned mylink_;

        // Constructors and destructor

        // Assignment operator(s)

        // Private member functions
	int unpackData(unsigned short * buf, bool ptflag);
        // Private const member functions

        // Data members

        // Static data members
	const static unsigned size_ = 3;
        // Inline function definitions

};

#endif

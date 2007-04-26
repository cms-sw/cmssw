#if !defined(CSCTFTBRAWFORMAT_CSCTFTBPTDATA_H)
#define CSCTFTBRAWFORMAT_CSCTFTBPTDATA_H
// -*- C++ -*-
//
// Package:     CSCTFTBRawFormat
// Module:      CSCTFTBPTData
// 
// Description: Header file for PT Data word
//
// Implementation:
//     <Notes on implementation>
//
// Author:      Lindsey Gray
// Created:     13.1.2005
//
// $Id: CSCTFTBPTData.h,v 1.2 2006/06/22 14:46:05 lgray Exp $
//
// Revision History
// $Log: CSCTFTBPTData.h,v $
// Revision 1.2  2006/06/22 14:46:05  lgray
// Forced commit of all code
//
// Revision 1.1  2006/06/22 00:34:18  lgray
// Moved all data format classes here. Removed old Packages from nightly
//
// Revision 1.1  2006/02/22 23:15:49  lgray
// First commit of test beam data format from UF
//
// Revision 1.2  2005/05/10 21:57:21  lgray
// Bugfixes, stability issues fixed
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

// External package include files

// STL classes

// Forward declarations

class CSCTFTBPTData 
{

// Friend classses and functions

// Public part
   public:
        // Constants, enums and typedefs

        // Constructors and destructor
        CSCTFTBPTData();
        CSCTFTBPTData(const CSCTFTBPTData&);

	//~CSCTFTBPTData();

        // Member functions
        static int size() {return size_;};

	CSCTFTBPTData operator=(const CSCTFTBPTData&);

        // Const member functions

	/// Pt LUT output of muon specified in header
	/// FR bit must select between 2 bytes
        unsigned int ptLUT(int fr) const;

        /// pretty-print
        friend std::ostream & operator<<(std::ostream & stream, const CSCTFTBPTData &);
        // Static member functions

// Protected part
   protected:

        // Protected member functions

        // Protected const member functions

// Private part
   private:

	//frame 1
	unsigned front_pt_    :8;
	unsigned rear_pt_     :8;

        // Constructors and destructor

        // Assignment operator(s)

        // Private member functions

        // Private const member functions

        // Data members

        // Static data members
	const static unsigned size_ = 1; // 16 bit words
        // Inline function definitions

};

#endif

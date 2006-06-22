#if !defined(CSCTFTBRAWFORMAT_CSCTFTBFRONTHEADER_H)
#define CSCTFTBRAWFORMAT_CSCTFTBFRONTHEADER_H
// -*- C++ -*-
//
// Package:     CSCTFTBRawFormat
// Module:      CSCTFTBFrontHeader
// 
// Description: Header file for SP Front Block Header class
//
// Implementation:
//     <Notes on implementation>
//
// Author:      Lindsey Gray
// Created:     13.1.2005
//
// $Id: CSCTFTBFrontHeader.h,v 1.1 2006/02/22 23:15:49 lgray Exp $

// System include files
#include <iostream>
// Package include files

// External package include files

// STL classes

// Forward declarations

class CSCTFTBFrontHeader 
{

// Friend classses and functions

// Public part
   public:
        // Constants, enums and typedefs

        // Constructors and destructor
        CSCTFTBFrontHeader();
	CSCTFTBFrontHeader(const CSCTFTBFrontHeader&);
        
	~CSCTFTBFrontHeader();

        // Member functions
  
        // Const member functions
	
	/// Valid Pattern word (15 links/bits)
	unsigned validPattern() const {return  vld_patrn_;}

	/// Valid Pattern for given link
	unsigned getVPBit(unsigned int frontFPGA, unsigned int link) const;

	/// Synch Error word (15 links/bits)
	unsigned synchError() const {return  synch_err_;}

	/// Synch Error for given link
	unsigned getSEBit(unsigned int frontFPGA, unsigned int link) const;

	/// Simple data integrity check
	bool check() const { return (zero1_==0 && zero2_==0);}

	/// size of data bank in 16-bit words
        static unsigned size() {return size_;};

        /// pretty-print
        friend std::ostream & operator<<(std::ostream & stream, const CSCTFTBFrontHeader &);
	friend class CSCTFTBFrontBlock;
// Private part
   private:

	//frame 1
	unsigned vld_patrn_  :15;
	unsigned zero1_      :1; // always zero
	//frame 2
	unsigned synch_err_  :15;
	unsigned zero2_      :1; // always zero

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

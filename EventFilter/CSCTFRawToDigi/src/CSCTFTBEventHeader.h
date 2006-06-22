#ifndef CSCTFTBRAWFORMAT_CSCTFTBEVENTHEADER_H
#define CSCTFTBRAWFORMAT_CSCTFTBEVENTHEADER_H
// -*- C++ -*-
//
// Package:     CSCTFTBRawFormat
// Module:      CSCTFTBEventHeader
// 
// Description: Header file for SP VME Header class,
//              rewritten to use bitfields
// Implementation:
//     <Notes on implementation>
//
// Author:      Darin Acosta / Holger Stoeck / Lindsey Gray
// Created:     13.1.2004
//
// $Id: CSCTFTBEventHeader.h,v 1.1 2006/06/22 00:34:18 lgray Exp $
//
// Revision History
// $Log: CSCTFTBEventHeader.h,v $
// Revision 1.1  2006/06/22 00:34:18  lgray
// Moved all data format classes here. Removed old Packages from nightly
//
// Revision 1.1  2006/02/22 23:15:49  lgray
// First commit of test beam data format from UF
//
// Revision 1.1  2005/02/14 20:59:46  lgray
// First Commit from UF
//
// Revision 1.10  2004/05/17 15:20:32  tfcvs
// DEA: expand header for TB 2004
//
// Revision 1.9  2003/08/27 22:08:07  tfcvs
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

class CSCTFTBEventHeader 
{

// Friend classses and functions

// Public part
 public:
        // Constants, enums and typedefs

        // Constructors and destructor
        CSCTFTBEventHeader();
        CSCTFTBEventHeader(const CSCTFTBEventHeader&);

	~CSCTFTBEventHeader();

        // Member functions

        // Const member functions
        /// L1A number
	int getLvl1num() const {return ((l1a_lsb_)|(l1a_msb_<<12));}
	
	/// BX number (for first bx read out)
	int getBXnum() const {return bunch_cntr_;}
	
	/// Zero Supp word
	unsigned int getZeroSupp() const {return zero_supp_;}
	
	/// DT data flag word
	unsigned int getActiveDT() const {return act_DT_;}
	
	/// ME1 data flag words
	unsigned int getActiveME1a() const {return getActiveFrontFPGA1();}
	unsigned int getActiveME1b() const {return getActiveFrontFPGA2();}
	unsigned int getActiveFrontFPGA1() const {return (act_ffpga_&1);}
	unsigned int getActiveFrontFPGA2() const {return ((act_ffpga_&2)>>1);}
	
	/// ME2 data flag words
	unsigned int getActiveME2() const {return getActiveFrontFPGA3();}
	unsigned int getActiveFrontFPGA3() const {return ((act_ffpga_&4)>>2);}

	/// ME3 data flag words
	unsigned int getActiveME3() const {return getActiveFrontFPGA4();}
	unsigned int getActiveFrontFPGA4() const {return ((act_ffpga_&8)>>3);}

	/// ME4 data flag words
	unsigned int getActiveME4() const {return getActiveFrontFPGA4();}
	unsigned int getActiveFrontFPGA5() const {return ((act_ffpga_&10)>>4);}

	/// Overall Active Front FPGA word
	unsigned int getActiveFront() const {return act_ffpga_;}

	/// Calculate number of active front FPGAs.
	unsigned int getNumActiveFront() const;

	/// PT LUT Spy flag 0 = no track, 1 = 1st track, 2 = 2nd, 3 = 3rd
	unsigned int getPtLutSpy() const { return pt_lut_spy_;}

        /// number of links to read out per MPC connection
	int numLinks() const {return nLinks_;}
	
	/// number of MPC connections to read out
	int numMPC() const {return nMPC_;}

	/// number of BX read out per link
	int numBX() const {return num_bx_;}
	
	/// simple data integrity check
	bool check() const;
	
        // Static member functions
	static unsigned size() {return size_;}
                
        /// pretty-print
        friend std::ostream & operator<<(std::ostream & stream, const CSCTFTBEventHeader &);
	
	/// filled by EventData
	friend class CSCTFTBEventData;
 private:
	//frame 1
	unsigned num_bx_     :3;
	unsigned zero_supp_  :1;
	unsigned act_ffpga_  :5;
	unsigned act_DT_     :1;
	unsigned pt_lut_spy_ :2;
	unsigned key1_       :4; // always 0xf
	//frame 2		
	unsigned bunch_cntr_ :12;
	unsigned key2_       :4; // always 0xf
	//frame 3
	unsigned l1a_lsb_    :12;
	unsigned key3_       :4; // always 0xf
	//frame 4
	unsigned l1a_msb_    :12;
	unsigned key4_       :4; // always 0xf

	static const unsigned size_ = 4;
	static const unsigned nLinks_ = 3;
	static const unsigned nMPC_ = 5;
};

#endif

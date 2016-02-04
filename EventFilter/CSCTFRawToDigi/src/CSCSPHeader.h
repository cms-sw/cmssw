#ifndef CSCSPHeader_h
#define CSCSPHeader_h

#include <string.h> // memcpy

class CSCSPHeader {
private:
	/////////////
	unsigned sp_l1a_low    : 12;
	unsigned header_mark_1 : 4;  // constant, should be 1001 = 0x9

	unsigned sp_l1a_high   : 12;
	unsigned header_mark_2 : 4;  // constant, should be 1001 = 0x9

	unsigned zero_1        : 12; // constant, should be 0
	unsigned header_mark_3 : 4;  // constant, should be 1001 = 0x9

	unsigned sp_bxn        : 12;
	unsigned header_mark_4 : 4;  // constant, should be 1001 = 0x9

	/////////////
	unsigned zero_2        : 12; // constant, should be 0
	unsigned header_mark_5 : 4;  // constant, should be 1010 = 0xA

	unsigned sp_slot_number     : 5; // SP_PADR, or physical address, or slot number
	unsigned sp_ersv            : 3; // event record structure version
	unsigned sp_trigger_sector  : 4; // SP Trigger Sector 1, 2, 3, 4, 5, 6 for +Z EMU side and 7, 8, 9, 10, 11, 12 for -Z EMU side

	unsigned header_mark_6      : 4; // constant, should be 1010 = 0xA
/*
	unsigned warning_overflow : 1; // FMM: L1A rate is too high
	unsigned out_of_sync      : 1; // FMM: L1 Reset is required
	unsigned busy             : 1; // FMM: L1A rate higher than SP can handle - start flushing events
	unsigned ready            : 1; // FMM: L1A rate is fine
	unsigned fa_out_of_sync   : 1; // FMM: FA out of sync: one of Front_FPGA's L1A doesn't match DDU_FPGA L1A
	unsigned sp_out_of_sync   : 1; // FMM: SP out of sync: SP_FPGA's L1A doesn't match DDU_FPGA L1A
	unsigned spare_1          : 1; // not used yet
	unsigned zero_3           : 5; // constant, should be 0
	unsigned header_mark_7    : 4; // constant, should be 1010 = 0xA
*/
	unsigned fmm_status       : 6; // see FMM above
	unsigned ddm              : 1; // readout mode: 0/1 = DDU/VME
	unsigned zero_3           : 5; // constant, should be 0
	unsigned header_mark_7    : 4; // constant, should be 1010 = 0xA

/*
	// DD/CSR_DFC: DAQ FIFO Configuration
	unsigned tbin             : 3; // sp record may contain up to 7 consequetive BXs
	unsigned zero_suppression : 1; // zero suppression: if set - put only valid LCTs, stubs, and track in the record
	unsigned f1_active        : 1; // read(1)/suppress(0) F1 record in the event
	unsigned f2_active        : 1; // read(1)/suppress(0) F2 record in the event
	unsigned f3_active        : 1; // read(1)/suppress(0) F3 record in the event
	unsigned f4_active        : 1; // read(1)/suppress(0) F4 record in the event
	unsigned f5_active        : 1; // read(1)/suppress(0) F5 record in the event
	unsigned dt_active        : 1; // read(1)/suppress(0) DT record in the event
	unsigned sp_active        : 1; // read(1)/suppress(0) SP record in the event
	unsigned skip             : 1; // event is skiped, only this header + trailer exist in the record
	unsigned header_mark_8    : 4; // constant, should be 1010 = 0xA
*/
	// DD/CSR_DFC: DAQ FIFO Configuration
	unsigned csr_dfc          : 11;//
	unsigned skip             : 1; // event is skiped, only this header + trailer exist in the record
	unsigned header_mark_8    : 4; // constant, should be 1010 = 0xA

	friend class CSCTFPacker;

public:
	bool check(void) const {
		return header_mark_1!=0x9 || header_mark_2!=0x9 || header_mark_3!=0x9 || header_mark_4!=0x9
			|| header_mark_5!=0xA || header_mark_6!=0xA || header_mark_7!=0xA || header_mark_8!=0xA
			|| zero_1!=0 || zero_2!=0 || zero_3!=0;
	}

	unsigned int BXN(void) const throw() { return  sp_bxn; }
	unsigned int L1A(void) const throw() { return (sp_l1a_high<<12)|sp_l1a_low; }

	unsigned int slot  (void) const throw() { return sp_slot_number; }
	unsigned int trigger_sector(void) const throw() { return sp_trigger_sector; }
	// following two functions are kept for compatibility with earlier versions of TF data format:
	unsigned int sector(void) const throw() {
		if(sp_ersv<2) return sp_trigger_sector&0x7;
		else return (sp_trigger_sector<=6?sp_trigger_sector:sp_trigger_sector-6);
	}
	unsigned int endcap(void) const throw() {
		if(sp_ersv<2) return sp_trigger_sector&0x8;
		else return (sp_trigger_sector<=6?1:0);
	}

	enum FMM {WOF=1,OSY=2,BUZY=4,READY=8,FA_OSY=16,SP_OSY=32};
	unsigned int status(void) const throw() { return fmm_status; }

	unsigned int nTBINs(void) const throw() { return csr_dfc&0x7; }

	bool suppression(void) const throw() { return csr_dfc&0x8; }

	enum ACTIVE {F1=1,F2=2,F3=4,F4=8,F5=16,DT=32,SP=64};
	unsigned int active(void) const throw() { return csr_dfc>>4; }

	bool empty(void) const throw() { return skip; }

	int  format_version(void) const throw() { return sp_ersv; }
	bool ddu_readout   (void) const throw() { return ddm; }

	bool unpack(const unsigned short *&buf) throw() { memcpy((void*)this,buf,8*sizeof(short)); buf+=8; return check(); }

	CSCSPHeader(void){}
};

#endif

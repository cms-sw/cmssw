#ifndef CSCDCCExaminer_h
#define CSCDCCExaminer_h
#include <set>
#include <map>
#include <vector>
#include <string>
#include <iostream>

/** DCC identifier type */
typedef int32_t DCCIdType;

/** DDU identifier type */
typedef int16_t DDUIdType;

/** CSC identifier type */
typedef int32_t CSCIdType;

/** Examiner status and mask type */
typedef uint32_t ExaminerMaskType;
typedef uint32_t ExaminerStatusType;

/** Format Error individual named flags */
typedef enum FormatErrorFlag {
  ANY_ERRORS                                          = 0,
  DDU_TRAILER_MISSING                                 = 1,
  DDU_HEADER_MISSING                                  = 2,
  DDU_CRC_ERROR                                       = 3,
  DDU_WORD_COUNT_ERROR                                = 4,
  DMB_TRAILER_MISSING                                 = 5,
  DMB_HEADER_MISSING                                  = 6,
  ALCT_TRAILER_MISSING                                = 7,
  ALCT_HEADER_MISSING                                 = 8,
  ALCT_WORD_COUNT_ERROR                               = 9,
  ALCT_CRC_ERROR                                      = 10,
  ALCT_TRAILER_BIT_ERROR                              = 11,
  TMB_TRAILER_MISSING                                 = 12,
  TMB_HEADER_MISSING                                  = 13,
  TMB_WORD_COUNT_ERROR                                = 14,
  TMB_CRC_ERROR                                       = 15,
  CFEB_WORD_COUNT_PER_SAMPLE_ERROR                    = 16,
  CFEB_SAMPLE_COUNT_ERROR                             = 17,
  CFEB_CRC_ERROR                                      = 18,
  DDU_EVENT_SIZE_LIMIT_ERROR                          = 19,
  C_WORDS                                             = 20,
  ALCT_DAV_ERROR                                      = 21,
  TMB_DAV_ERROR                                       = 22,
  CFEB_DAV_ERROR                                      = 23,
  DMB_ACTIVE_ERROR                                    = 24,
  DCC_TRAILER_MISSING                                 = 25,
  DCC_HEADER_MISSING                                  = 26,
  DMB_DAV_VS_DMB_ACTIVE_MISMATCH_ERROR                = 27,
  EXTRA_WORDS_BETWEEN_DDU_HEADER_AND_FIRST_DMB_HEADER = 28
};

/** CSC Payload individual named flags */
typedef enum CSCPayloadFlag {
  CFEB1_ACTIVE = 0,
  CFEB2_ACTIVE = 1,
  CFEB3_ACTIVE = 2,
  CFEB4_ACTIVE = 3,
  CFEB5_ACTIVE = 4,
  ALCT_DAV     = 5,
  TMB_DAV      = 6,
  CFEB1_DAV    = 7,
  CFEB2_DAV    = 8,
  CFEB3_DAV    = 9,
  CFEB4_DAV    = 10,
  CFEB5_DAV    = 11
};

/** CSC Status individual named flags */
typedef enum CSCStatusFlag {
  ALCT_FIFO_FULL           = 0,
  TMB_FIFO_FULL            = 1,
  CFEB1_FIFO_FULL          = 2,
  CFEB2_FIFO_FULL          = 3,
  CFEB3_FIFO_FULL          = 4,
  CFEB4_FIFO_FULL          = 5,
  CFEB5_FIFO_FULL          = 6,
  ALCT_START_TIMEOUT       = 7,
  TMB_START_TIMEOUT        = 8,
  CFEB1_START_TIMEOUT      = 9,
  CFEB2_START_TIMEOUT      = 10,
  CFEB3_START_TIMEOUT      = 11,
  CFEB4_START_TIMEOUT      = 12,
  CFEB5_START_TIMEOUT      = 13,
  ALCT_END_TIMEOUT         = 14,
  TMB_END_TIMEOUT          = 15,
  CFEB1_END_TIMEOUT        = 16,
  CFEB2_END_TIMEOUT        = 17,
  CFEB3_END_TIMEOUT        = 18,
  CFEB4_END_TIMEOUT        = 19,
  CFEB5_END_TIMEOUT        = 20,
  CFEB_ACTIVE_DAV_MISMATCH = 21,
  B_WORDS_FOUND            = 22
};

class CSCDCCExaminer {
public:
	const uint16_t nERRORS, nWARNINGS, nPAYLOADS, nSTATUSES;

private:
	std::vector<char*> sERROR,  sWARNING, sERROR_,  sWARNING_, sDMBExpectedPayload, sDMBEventStaus;
	ExaminerStatusType bERROR,  bWARNING;
	bool               fERROR  [29];//[nERRORS];
	bool               fWARNING[5]; //[nWARNINGS];

	std::set<CSCIdType>      fCHAMB_ERR[29]; // Set of chambers which contain particular error
	std::set<CSCIdType>      fCHAMB_WRN[5];  // Set of chambers which contain particular warning
	std::map<CSCIdType,ExaminerStatusType> bCHAMB_ERR;     // chamber <=> errors in bits
	std::map<CSCIdType,ExaminerStatusType> bCHAMB_WRN;     // chamber <=> errors in bits
	std::map<CSCIdType,ExaminerStatusType> bCHAMB_PAYLOAD; //
	std::map<CSCIdType,ExaminerStatusType> bCHAMB_STATUS;  //
	std::map<DDUIdType,ExaminerStatusType> bDDU_ERR;       // ddu     <-> errors in bits
	std::map<DDUIdType,ExaminerStatusType> bDDU_WRN;       // ddu     <-> errors in bits

	class OStream : public std::ostream {
	private:
		class buffer : public std::streambuf{};
		buffer     buff;
		std::streambuf *stream;
		std::streambuf *null;
		std::string     name;

	public:
		void show(void){ rdbuf(stream); }
		void hide(void){ rdbuf(null);   }
		void sign(std::string nm)     { name=nm; }
		void sign(const char *nm){ name=nm; }

		void redirect(std::ostream &str){
			stream = str.rdbuf(); tie(&str);
			if( rdbuf() != null ) rdbuf(stream);
		}

		template<class T> std::ostream& operator<<(const T& val){
			return (*(std::ostream*)this)<<name<<val;
		}

		OStream(void):std::ostream(std::cout.rdbuf()),buff(),stream(std::cout.rdbuf()),null(&buff),name(""){}
		OStream(std::ostream &str):std::ostream(str.rdbuf()),buff(),stream(str.rdbuf()),null(&buff),name(""){}
	};

	OStream cout, cerr;

	CSCIdType currentChamber;       // ( (CrateNumber<<4) + DMBslot ) specifies chamber

	const uint16_t *buf_2, *buf_1, *buf0, *buf1, *buf2;
		  uint16_t tmpbuf[16];

	bool fDCC_Header;
	bool fDCC_Trailer;
	bool fDDU_Header;
	bool fDDU_Trailer;
	bool fDMB_Header;
	bool fDMB_Trailer;
	bool fALCT_Header;
	bool fTMB_Header;
	bool fTMB_Format2007;
	bool fALCT_Format2007;

	bool uniqueALCT, uniqueTMB; // Do not merge two DMBs if Trailer of the first and Header of the second are lost

	bool DAV_ALCT; // ...
	bool DAV_TMB;  // Check if DAV bits lie
	int  DAV_CFEB; // ...
	int  DAV_DMB;  // ...
	int  DMB_Active, nDMBs;  // ...

public:
	uint32_t cntDDU_Headers;
	uint32_t cntDDU_Trailers;
	std::map<CSCIdType,uint32_t> cntCHAMB_Headers;
	std::map<CSCIdType,uint32_t> cntCHAMB_Trailers;

private:
        void clear();
        void zeroCounts();
        /// checks DAV_ALCT, DAV_TMB, and DAV_CFEB
        void checkDAVs();
        void checkTriggerHeadersAndTrailers();
	uint32_t DDU_WordsSinceLastHeader;
	uint32_t DDU_WordCount;
	uint32_t DDU_WordMismatch_Occurrences;
	uint32_t DDU_WordsSinceLastTrailer;

	uint32_t ALCT_WordsSinceLastHeader;
	uint32_t ALCT_WordCount;
	uint32_t ALCT_WordsExpected;

	uint32_t TMB_WordsSinceLastHeader;
	uint32_t TMB_WordCount;
	uint32_t TMB_WordsExpected;
	uint32_t TMB_Tbins;
	uint32_t TMB_WordsExpectedCorrection;
	uint32_t TMB_Firmware_Revision;

	uint32_t CFEB_SampleWordCount;
	uint32_t CFEB_SampleCount;
	uint32_t CFEB_BSampleCount;

	bool checkCrcALCT;
	uint32_t ALCT_CRC;
	bool checkCrcTMB;
	uint32_t TMB_CRC;
	bool checkCrcCFEB;
	uint32_t CFEB_CRC;

	bool  modeDDUonly;
	DDUIdType sourceID;
	ExaminerMaskType examinerMask;

	//int headerDAV_Active; // Obsolete since 16.09.05

	// data blocks:
	std::map<DDUIdType,const uint16_t*>                  dduBuffers; // < DDUsourceID, pointer >
	std::map<DDUIdType,std::map<CSCIdType,const uint16_t*> > dmbBuffers; // < DDUsourceID, < DMBid, pointer > >
	std::map<DDUIdType,uint32_t>                  dduOffsets; // < DDUsourceID, pointer_offset >
	std::map<DDUIdType,std::map<CSCIdType,uint32_t> > dmbOffsets; // < DDUsourceID, < DMBid, pointer_offset > >
	std::map<DDUIdType,uint32_t>                  dduSize; // < DDUsourceID, block_size >
	std::map<DDUIdType,std::map<CSCIdType,uint32_t> > dmbSize; // < DDUsourceID, < DMBid, block_size > >
	const uint16_t *buffer_start;

public:
	OStream& output1(void){ return cout; }
	OStream& output2(void){ return cerr; }

	int32_t check(const uint16_t* &buffer, int32_t length);

	void setMask(ExaminerMaskType mask) {examinerMask=mask;}
        ExaminerMaskType getMask() const {return examinerMask;}

	ExaminerStatusType errors  (void) const { return bERROR;   }
	ExaminerStatusType warnings(void) const { return bWARNING; }

	const char* errName(int num) const { if(num>=0&&num<nERRORS)   return sERROR[num];   else return ""; }
	const char* wrnName(int num) const { if(num>=0&&num<nWARNINGS) return sWARNING[num]; else return ""; }

	const char* errorName  (int num) const { if(num>=0&&num<nERRORS)   return sERROR_[num];   else return ""; }
	const char* warningName(int num) const { if(num>=0&&num<nWARNINGS) return sWARNING_[num]; else return ""; }

	const char* payloadName(int num) const { if(num>=0&&num<nPAYLOADS) return sDMBExpectedPayload[num]; else return ""; }
	const char* statusName (int num) const { if(num>=0&&num<nSTATUSES) return sDMBEventStaus     [num]; else return ""; }

	bool error  (int num) const { if(num>=0&&num<nERRORS)   return fERROR  [num]; else return 0; }
	bool warning(int num) const { if(num>=0&&num<nWARNINGS) return fWARNING[num]; else return 0; }

	std::set<CSCIdType> chambersWithError  (int num) const { if(num>=0&&num<nERRORS)   return fCHAMB_ERR[num]; else return std::set<int>(); }
	std::set<CSCIdType> chambersWithWarning(int num) const { if(num>=0&&num<nWARNINGS) return fCHAMB_WRN[num]; else return std::set<int>(); }

	ExaminerStatusType payloadForChamber(CSCIdType chamber) const {
		std::map<CSCIdType,ExaminerStatusType>::const_iterator item = bCHAMB_PAYLOAD.find(chamber);
		if( item != bCHAMB_PAYLOAD.end() ) return item->second; else return 0;
	}

	ExaminerStatusType statusForChamber(CSCIdType chamber) const {
		std::map<CSCIdType,ExaminerStatusType>::const_iterator item = bCHAMB_STATUS.find(chamber);
		if( item != bCHAMB_STATUS.end() ) return item->second; else return 0;
	}

	ExaminerStatusType errorsForChamber(CSCIdType chamber) const {
		std::map<CSCIdType,ExaminerStatusType>::const_iterator item = bCHAMB_ERR.find(chamber);
		if( item != bCHAMB_ERR.end() ) return item->second; else return 0;
	}

	ExaminerStatusType warningsForChamber(CSCIdType chamber) const {
		std::map<CSCIdType,ExaminerStatusType>::const_iterator item = bCHAMB_WRN.find(chamber);
		if( item != bCHAMB_WRN.end() ) return item->second; else return 0;
	}

	ExaminerStatusType errorsForDDU(DDUIdType dduSourceID) const {
		std::map<DDUIdType,ExaminerStatusType>::const_iterator item = bDDU_ERR.find(dduSourceID);
		if( item != bDDU_ERR.end() ) return item->second; else return 0;
	}
	ExaminerStatusType warningsForDDU(DDUIdType dduSourceID) const {
		std::map<DDUIdType,ExaminerStatusType>::const_iterator item = bDDU_WRN.find(dduSourceID);
		if( item != bDDU_WRN.end() ) return item->second; else return 0;
	}
	std::vector<DDUIdType> listOfDDUs(void) const {
		std::vector<DDUIdType> DDUs;
		std::map<DDUIdType,ExaminerStatusType>::const_iterator item = bDDU_ERR.begin();
		while( item != bDDU_ERR.end() ){ DDUs.push_back(item->first); item++; }
		return DDUs;
	}

	std::map<DDUIdType,ExaminerStatusType> errorsDetailedDDU  (void) const { return bDDU_ERR; }

	std::map<CSCIdType,ExaminerStatusType> errorsDetailed  (void) const { return bCHAMB_ERR; }
	std::map<CSCIdType,ExaminerStatusType> warningsDetailed(void) const { return bCHAMB_WRN; }
	std::map<CSCIdType,ExaminerStatusType> payloadDetailed (void) const { return bCHAMB_PAYLOAD; }
	std::map<CSCIdType,ExaminerStatusType> statusDetailed  (void) const { return bCHAMB_STATUS; }

	

	void crcALCT(bool enable);
	void crcTMB (bool enable);
	void crcCFEB(bool enable);

	void modeDDU(bool enable);

	DDUIdType dduSourceID(void){ return sourceID; }

	std::map<DDUIdType,const uint16_t*>                  DDU_block(void) const { return dduBuffers; }
	std::map<DDUIdType,std::map<CSCIdType,const uint16_t*> > DMB_block(void) const { return dmbBuffers; }

	std::map<DDUIdType,uint32_t>                  DDU_ptrOffsets(void) const { return dduOffsets; }
	std::map<DDUIdType,std::map<CSCIdType,uint32_t> > DMB_ptrOffsets(void) const { return dmbOffsets; }

	std::map<DDUIdType,uint32_t>                  DDU_size(void) const { return dduSize; }
	std::map<DDUIdType,std::map<CSCIdType,uint32_t> > DMB_size(void) const { return dmbSize; }

	CSCDCCExaminer(ExaminerMaskType mask=0x1); 
	~CSCDCCExaminer(void){}
};

#endif

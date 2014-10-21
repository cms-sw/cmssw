#ifndef CSCDCCExaminer_h
#define CSCDCCExaminer_h
#include <set>
#include <map>
#include <vector>
#include <string>
#ifdef LOCAL_UNPACK
#include <iostream>
#else
#include <ostream>
#endif

#include "DataFormats/CSCDigi/interface/CSCDCCFormatStatusDigi.h"

class CSCDCCExaminer {
public:
	const uint16_t nERRORS, nWARNINGS, nPAYLOADS, nSTATUSES;

private:
	std::vector<const char*> sERROR,  sWARNING, sERROR_,  sWARNING_, sDMBExpectedPayload, sDMBEventStaus;
	ExaminerStatusType bERROR,  bWARNING;
        ExaminerStatusType bSUM_ERROR,  bSUM_WARNING;	// Summary flags for errors and warnings
	bool               fERROR  [29];//[nERRORS];
	bool               fWARNING[5]; //[nWARNINGS];
        bool               fSUM_ERROR  [29];//[nERRORS];
        bool               fSUM_WARNING[5]; //[nWARNINGS];

	std::set<CSCIdType>      fCHAMB_ERR[29]; // Set of chambers which contain particular error
	std::set<CSCIdType>      fCHAMB_WRN[5];  // Set of chambers which contain particular warning
	std::map<CSCIdType,ExaminerStatusType> bCHAMB_ERR;     // chamber <=> errors in bits
	std::map<CSCIdType,ExaminerStatusType> bCHAMB_WRN;     // chamber <=> errors in bits
	std::map<CSCIdType,ExaminerStatusType> bCHAMB_PAYLOAD; //
	std::map<CSCIdType,ExaminerStatusType> bCHAMB_STATUS;  //
	std::map<DDUIdType,ExaminerStatusType> bDDU_ERR;       // ddu     <-> errors in bits
	std::map<DDUIdType,ExaminerStatusType> bDDU_WRN;       // ddu     <-> errors in bits

#ifdef LOCAL_UNPACK
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

	OStream COUT, CERR;
#endif

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
        bool fFormat2013;

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
        void sync_stats();
        /// checks DAV_ALCT, DAV_TMB, and DAV_CFEB
        void checkDAVs();
        void checkTriggerHeadersAndTrailers();

        inline int scanbuf(const uint16_t* &buf, int32_t length, uint16_t sig, uint16_t mask=0xFFFF);

	uint32_t DDU_WordsSinceLastHeader;
	uint32_t DDU_WordCount;
	uint32_t DDU_WordMismatch_Occurrences;
	uint32_t DDU_WordsSinceLastTrailer;
        
	uint32_t ALCT_WordsSinceLastHeader;
        uint32_t ALCT_WordsSinceLastHeaderZeroSuppressed; 
	uint32_t ALCT_WordCount;
	uint32_t ALCT_WordsExpected;
        uint32_t ALCT_ZSE;       /// check zero suppression mode
        uint32_t nWG_round_up;   /// to decode if zero suppression enabled

	uint32_t TMB_WordsSinceLastHeader;
	uint32_t TMB_WordCount;
	uint32_t TMB_WordsExpected;
	uint32_t TMB_Tbins;
	uint32_t TMB_WordsRPC;
	uint32_t TMB_Firmware_Revision;
  	uint32_t DDU_Firmware_Revision;

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

#ifdef LOCAL_UNPACK
	OStream& output1(void){ return COUT; }
	OStream& output2(void){ return CERR; }
#endif

	int32_t check(const uint16_t* &buffer, int32_t length);

	void setMask(ExaminerMaskType mask) {examinerMask=mask;}
        ExaminerMaskType getMask() const {return examinerMask;}

	ExaminerStatusType errors  (void) const { return bSUM_ERROR;   }
	ExaminerStatusType warnings(void) const { return bSUM_WARNING; }

	const char* errName(int num) const { if(num>=0&&num<nERRORS)   return sERROR[num];   else return ""; }
	const char* wrnName(int num) const { if(num>=0&&num<nWARNINGS) return sWARNING[num]; else return ""; }

	const char* errorName  (int num) const { if(num>=0&&num<nERRORS)   return sERROR_[num];   else return ""; }
	const char* warningName(int num) const { if(num>=0&&num<nWARNINGS) return sWARNING_[num]; else return ""; }

	const char* payloadName(int num) const { if(num>=0&&num<nPAYLOADS) return sDMBExpectedPayload[num]; else return ""; }
	const char* statusName (int num) const { if(num>=0&&num<nSTATUSES) return sDMBEventStaus     [num]; else return ""; }

	bool error  (int num) const { if(num>=0&&num<nERRORS)   return fSUM_ERROR  [num]; else return 0; }
	bool warning(int num) const { if(num>=0&&num<nWARNINGS) return fSUM_WARNING[num]; else return 0; }

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
                /// Print (for debugging, to be removed)
                
                // for(item =bCHAMB_ERR.begin() ; item !=bCHAMB_ERR.end() ; item++)
                //std::cout << " Ex-errors: " << std::hex << (*item).second << std::dec << std::endl;

                item = bCHAMB_ERR.find(chamber);
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

        bool isDDUmode() {return modeDDUonly;};

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

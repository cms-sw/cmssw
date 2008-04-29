#ifndef CSCDCCExaminer_h
#define CSCDCCExaminer_h
#include <set>
#include <map>
#include <vector>
#include <string>
#include <iostream>

class CSCDCCExaminer {
public:
	const unsigned short nERRORS, nWARNINGS;

private:
	std::vector<char*> sERROR,  sWARNING, sERROR_,  sWARNING_, sDMBExpectedPayload, sDMBEventStaus;
	long               bERROR,  bWARNING;
	bool               fERROR  [29];//[nERRORS];
	bool               fWARNING[5]; //[nWARNINGS];

	std::set<int>      fCHAMB_ERR[29]; // Set of chambers which contain particular error
	std::set<int>      fCHAMB_WRN[5];  // Set of chambers which contain particular warning
	std::map<int,long> bCHAMB_ERR;     // chamber <=> errors in bits
	std::map<int,long> bCHAMB_WRN;     // chamber <=> errors in bits
	std::map<int,long> bCHAMB_PAYLOAD; //
	std::map<int,long> bCHAMB_STATUS;  //
	std::map<int,long> bDDU_ERR;       // ddu     <-> errors in bits
	std::map<int,long> bDDU_WRN;       // ddu     <-> errors in bits

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

	int currentChamber;       // ( (CrateNumber<<4) + DMBslot ) specifies chamber

	const unsigned short *buf_2, *buf_1, *buf0, *buf1, *buf2;
		  unsigned short tmpbuf[16];

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
	long cntDDU_Headers;
	long cntDDU_Trailers;
	std::map<int,long> cntCHAMB_Headers;
	std::map<int,long> cntCHAMB_Trailers;

private:
	long DDU_WordsSinceLastHeader;
	long DDU_WordCount;
	long DDU_WordMismatch_Occurrences;
	long DDU_WordsSinceLastTrailer;

	long ALCT_WordsSinceLastHeader;
	long ALCT_WordCount;
	long ALCT_WordsExpected;

	long TMB_WordsSinceLastHeader;
	long TMB_WordCount;
	long TMB_WordsExpected;
	long TMB_Tbins;
	long TMB_WordsExpectedCorrection;

	long CFEB_SampleWordCount;
	long CFEB_SampleCount;
	long CFEB_BSampleCount;

	bool checkCrcALCT;
	unsigned long ALCT_CRC;
	bool checkCrcTMB;
	unsigned long TMB_CRC;
	bool checkCrcCFEB;
	unsigned long CFEB_CRC;

	bool  modeDDUonly;
	short sourceID;

	//int headerDAV_Active; // Obsolete since 16.09.05

	// data blocks:
	std::map<short,const unsigned short*>                  dduBuffers; // < DDUsourceID, pointer >
	std::map<short,std::map<short,const unsigned short*> > dmbBuffers; // < DDUsourceID, < DMBid, pointer > >
	std::map<short,unsigned long>                  dduOffsets; // < DDUsourceID, pointer_offset >
	std::map<short,std::map<short,unsigned long> > dmbOffsets; // < DDUsourceID, < DMBid, pointer_offset > >
	std::map<short,unsigned long>                  dduSize; // < DDUsourceID, block_size >
	std::map<short,std::map<short,unsigned long> > dmbSize; // < DDUsourceID, < DMBid, block_size > >
	const unsigned short *buffer_start;

public:
	OStream& output1(void){ return cout; }
	OStream& output2(void){ return cerr; }

	long check(const unsigned short* &buffer, long length);

	long errors  (void) const { return bERROR;   }
	long warnings(void) const { return bWARNING; }

	const char* errName(int num) const { if(num>=0&&num<nERRORS)   return sERROR[num];   else return ""; }
	const char* wrnName(int num) const { if(num>=0&&num<nWARNINGS) return sWARNING[num]; else return ""; }

	const char* errorName  (int num) const { if(num>=0&&num<nERRORS)   return sERROR_[num];   else return ""; }
	const char* warningName(int num) const { if(num>=0&&num<nWARNINGS) return sWARNING_[num]; else return ""; }

	const char* payloadName(int num) const { if(num>=0&&num<13) return sDMBExpectedPayload[num]; else return ""; }
	const char* statusName (int num) const { if(num>=0&&num<19) return sDMBEventStaus     [num]; else return ""; }

	bool error  (int num) const { if(num>=0&&num<nERRORS)   return fERROR  [num]; else return 0; }
	bool warning(int num) const { if(num>=0&&num<nWARNINGS) return fWARNING[num]; else return 0; }

	std::set<int> chambersWithError  (int num) const { if(num>=0&&num<nERRORS)   return fCHAMB_ERR[num]; else return std::set<int>(); }
	std::set<int> chambersWithWarning(int num) const { if(num>=0&&num<nWARNINGS) return fCHAMB_WRN[num]; else return std::set<int>(); }

	long payloadForChamber(int chamber) const {
		std::map<int,long>::const_iterator item = bCHAMB_PAYLOAD.find(chamber);
		if( item != bCHAMB_PAYLOAD.end() ) return item->second; else return 0;
	}

	long statusForChamber(int chamber) const {
		std::map<int,long>::const_iterator item = bCHAMB_STATUS.find(chamber);
		if( item != bCHAMB_STATUS.end() ) return item->second; else return 0;
	}

	long errorsForChamber(int chamber) const {
		std::map<int,long>::const_iterator item = bCHAMB_ERR.find(chamber);
		if( item != bCHAMB_ERR.end() ) return item->second; else return 0;
	}

	long warningsForChamber(int chamber) const {
		std::map<int,long>::const_iterator item = bCHAMB_WRN.find(chamber);
		if( item != bCHAMB_WRN.end() ) return item->second; else return 0;
	}

	long errorsForDDU(int dduSourceID) const {
		std::map<int,long>::const_iterator item = bDDU_ERR.find(dduSourceID);
		if( item != bDDU_ERR.end() ) return item->second; else return 0;
	}
	long warningsForDDU(int dduSourceID) const {
		std::map<int,long>::const_iterator item = bDDU_WRN.find(dduSourceID);
		if( item != bDDU_WRN.end() ) return item->second; else return 0;
	}
	std::vector<int> listOfDDUs(void) const {
		std::vector<int> DDUs;
		std::map<int,long>::const_iterator item = bDDU_ERR.begin();
		while( item != bDDU_ERR.end() ){ DDUs.push_back(item->first); item++; }
		return DDUs;
	}

	std::map<int,long> errorsDetailed  (void) const { return bCHAMB_ERR; }
	std::map<int,long> warningsDetailed(void) const { return bCHAMB_WRN; }

	void crcALCT(bool enable);
	void crcTMB (bool enable);
	void crcCFEB(bool enable);

	void modeDDU(bool enable);

	short dduSourceID(void){ return sourceID; }

	std::map<short,const unsigned short*>                  DDU_block(void) const { return dduBuffers; }
	std::map<short,std::map<short,const unsigned short*> > DMB_block(void) const { return dmbBuffers; }

	std::map<short,unsigned long>                  DDU_ptrOffsets(void) const { return dduOffsets; }
	std::map<short,std::map<short,unsigned long> > DMB_ptrOffsets(void) const { return dmbOffsets; }

	std::map<short,unsigned long>                  DDU_size(void) const { return dduSize; }
	std::map<short,std::map<short,unsigned long> > DMB_size(void) const { return dmbSize; }

	CSCDCCExaminer(void);
	~CSCDCCExaminer(void){}
};

#endif

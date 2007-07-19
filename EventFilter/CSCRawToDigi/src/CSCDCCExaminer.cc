#include "EventFilter/CSCRawToDigi/interface/CSCDCCExaminer.h"
#include <string>
#include <string.h>
#include <iomanip>
using namespace std;

void CSCDCCExaminer::crcALCT(bool enable){
  checkCrcALCT = enable;
  if( checkCrcALCT )
    sERROR[10] = "ALCT CRC Error                                   ";
  else
    sERROR[10] = "ALCT CRC Error ( disabled )                      ";
}

void CSCDCCExaminer::crcTMB(bool enable){
  checkCrcTMB = enable;
  if( checkCrcTMB )
    sERROR[15] = "TMB CRC Error                                    ";
  else
    sERROR[15] = "TMB CRC Error ( disabled )                       ";
}

void CSCDCCExaminer::crcCFEB(bool enable){
  checkCrcCFEB = enable;
  if( checkCrcCFEB )
    sERROR[18] = "CFEB CRC Error                                   ";
  else
    sERROR[18] = "CFEB CRC Error ( disabled )                      ";
}

void CSCDCCExaminer::modeDDU(bool enable){
  modeDDUonly = enable;
  if( modeDDUonly) {
    sERROR[25] = "DCC Trailer Missing                              ";
    sERROR[26] = "DCC Header Missing                               ";
  } else {
    sERROR[25] = "DCC Trailer Missing (disabled)                   ";
    sERROR[26] = "DCC Header Missing (disabled)                    ";
  }

}


CSCDCCExaminer::CSCDCCExaminer(void):nERRORS(27),nWARNINGS(5),sERROR(nERRORS),sWARNING(nWARNINGS),sERROR_(nERRORS),sWARNING_(nWARNINGS){
  cout.redirect(std::cout); cerr.redirect(std::cerr);

  sERROR[0] = " Any errors                                       ";
  sERROR[1] = " DDU Trailer Missing                              ";
  sERROR[2] = " DDU Header Missing                               ";
  sERROR[4] = " DDU Word Count Error                             ";
  sERROR[3] = " DDU CRC Error (not yet implemented)              ";
  sERROR[5] = " DMB Trailer Missing                              ";
  sERROR[6] = " DMB Header Missing                               ";
  sERROR[7] = " ALCT Trailer Missing                             ";
  sERROR[8] = " ALCT Header Missing                              ";
  sERROR[9] = " ALCT Word Count Error                            ";
  sERROR[10] = "ALCT CRC Error                                   ";
  sERROR[11] = "ALCT Trailer Bit Error                           ";
  // ^^^ This is due to seeing many events in ddu293 (also, some in ddu294)
  // with some bits in the 1st ALCT D-Header being lost. This causes a chain of errors:
  // - TMB Trailer is not identified and TMB word count mismatch occur when Trailer is found
  // - CFEB sample count is not reset on ALCT Trailer.
  // To merge all these errors in one,
  // the D-signature in the 1st ALCT Trailer will not be required for identifying the ALCT Trailer;
  // However, if these bits are found to be missing, ERROR[11] will be flagged.
  // This is just a temporary patch to make the output look less clattered.
  sERROR[12] = "TMB Trailer Missing                              ";
  sERROR[13] = "TMB Header Missing                               ";
  sERROR[14] = "TMB Word Count Error                             ";
  sERROR[15] = "TMB CRC Error                                    ";
  sERROR[16] = "CFEB Word Count Per Sample Error                 ";
  sERROR[17] = "CFEB Sample Count Error                          ";
  sERROR[18] = "CFEB CRC Error                                   ";
  sERROR[19] = "DDU Event Size Limit Error                       ";
  sERROR[20] = "C-Words                                          ";
  sERROR[21] = "ALCT DAV Error                                   ";
  sERROR[22] = "TMB DAV Error                                    ";
  sERROR[23] = "CFEB DAV Error                                   ";
  sERROR[24] = "DMB Active Error                                 ";
  sERROR[25] = "DCC Trailer Missing                              ";
  sERROR[26] = "DCC Header Missing                               ";

  //	sERROR[21] = "DDU Header vs. Trailer mismatch for DAV or Avtive"; // oboslete since 16.09.05

  sWARNING[0] = " Extra words between DDU Trailer and DDU Header ";
  sWARNING[1] = " CFEB B-Words                                   ";

  sERROR_[0] = " Any errors: 00";
  sERROR_[1] = " DDU Trailer Missing: 01";
  sERROR_[2] = " DDU Header Missing: 02";
  sERROR_[4] = " DDU Word Count Error: 04";
  sERROR_[3] = " DDU CRC Error (not yet implemented): 03";
  sERROR_[5] = " DMB Trailer Missing: 05";
  sERROR_[6] = " DMB Header Missing: 06";
  sERROR_[7] = " ALCT Trailer Missing: 07";
  sERROR_[8] = " ALCT Header Missing: 08";
  sERROR_[9] = " ALCT Word Count Error: 09";
  sERROR_[10] = "ALCT CRC Error: 10";
  sERROR_[11] = "ALCT Trailer Bit Error: 11";
  sERROR_[12] = "TMB Trailer Missing: 12";
  sERROR_[13] = "TMB Header Missing: 13";
  sERROR_[14] = "TMB Word Count Error: 14";
  sERROR_[15] = "TMB CRC Error: 15";
  sERROR_[16] = "CFEB Word Count Per Sample Error: 16";
  sERROR_[17] = "CFEB Sample Count Error: 17";
  sERROR_[18] = "CFEB CRC Error: 18";
  sERROR_[19] = "DDU Event Size Limit Error: 19";
  sERROR_[20] = "C-Words: 20";
  sERROR_[21] = "ALCT DAV Error: 21";
  sERROR_[22] = "TMB DAV Error: 22";
  sERROR_[23] = "CFEB DAV Error: 23";
  sERROR_[24] = "DMB Active Error: 24";
  sERROR_[25] = "DCC Trailer Missing: 25";
  sERROR_[26] = "DCC Header Missing: 26";
  //	sERROR_[21] = "DDU Header vs. Trailer mismatch for DAV or Avtive: 21"; // oboslete since 16.09.05

  sWARNING_[0] = " Extra words between DDU Trailer and DDU Header: 00";
  sWARNING_[1] = " CFEB B-Words: 01";

  fDCC_Header  = false;
  fDCC_Trailer = false;
  fDDU_Header  = false;
  fDDU_Trailer = false;
  fDMB_Header  = false;
  fDMB_Trailer = false;
  fALCT_Header = false;
  fTMB_Header  = false;

  cntDDU_Headers  = 0;
  cntDDU_Trailers = 0;
  cntCHAMB_Headers.clear();
  cntCHAMB_Trailers.clear();

  DAV_ALCT = false;
  DAV_TMB  = false;
  DAV_CFEB = 0;
  DAV_DMB  = 0;

  DDU_WordsSinceLastHeader     = 0;
  DDU_WordCount                = 0;
  DDU_WordMismatch_Occurrences = 0;
  DDU_WordsSinceLastTrailer    = 0;

  ALCT_WordsSinceLastHeader    = 0;
  ALCT_WordCount               = 0;
  ALCT_WordsExpected           = 0;

  TMB_WordsSinceLastHeader     = 0;
  TMB_WordCount                = 0;
  TMB_WordsExpected            = 0;
  TMB_Tbins                    = 0;
  TMB_WordsExpectedCorrection  = 0;

  CFEB_SampleWordCount         = 0;
  CFEB_SampleCount             = 0;
  CFEB_BSampleCount            = 0;

  checkCrcALCT = false; ALCT_CRC=0;
  checkCrcTMB  = false; TMB_CRC=0;
  checkCrcCFEB = false; CFEB_CRC=0;
  
  modeDDUonly = false;

  //headerDAV_Active = -1; // Trailer vs. Header check // Obsolete since 16.09.05


  bERROR   = 0;
  bWARNING = 0;
  bzero(fERROR,  sizeof(fERROR));
  bzero(fWARNING,sizeof(fWARNING));

  bCHAMB_ERR.clear();
  bCHAMB_WRN.clear();
  for(int err=0; err<nERRORS;   ++err) fCHAMB_ERR[err].clear();
  for(int wrn=0; wrn<nWARNINGS; ++wrn) fCHAMB_WRN[wrn].clear();

  buf_1 = &(tmpbuf[0]);
  buf0  = &(tmpbuf[4]);
  buf1  = &(tmpbuf[8]);
  buf2  = &(tmpbuf[12]);

  bzero(tmpbuf, sizeof(short)*16);
}

long CSCDCCExaminer::check(const unsigned short* &buffer, long length){
  if( length<=0 ) return -1;

  while( length>0 ){
    // == Store last 4 read buffers in pipeline-like memory (note that memcpy works quite slower!)
    buf_2 = buf_1;         //  This bufer was not needed so far
    buf_1 = buf0;
    buf0  = buf1;
    buf1  = buf2;
    buf2  = buffer;

    // check for too long event
    if(!fERROR[19] && DDU_WordsSinceLastHeader>50000 ){
      fERROR[19] = true;
      bERROR    |= 0x80000;
    }

    // increment counter of 64-bit words since last DDU Header
    // this counter is reset if DDU Header is found
    if ( fDDU_Header ) { ++DDU_WordsSinceLastHeader; }

    // increment counter of 64-bit words since last DDU Trailer
    // this counter is reset if DDU Trailer is found
    if ( fDDU_Trailer ) {++DDU_WordsSinceLastTrailer; }

    // increment counter of 16-bit words since last DMB*ALCT Header match
    // this counter is reset if ALCT Header is found right after DMB Header
    if ( fALCT_Header ) { ALCT_WordsSinceLastHeader = ALCT_WordsSinceLastHeader + 4; }

    // increment counter of 16-bit words since last DMB*TMB Header match
    // this counter is reset if TMB Header is found right after DMB Header or ALCT Trailer
    if ( fTMB_Header ) { TMB_WordsSinceLastHeader = TMB_WordsSinceLastHeader + 4; }

    // increment counter of 16-bit words since last of DMB Header, ALCT Trailer, TMB Trailer,
    // CFEB Sample Trailer, CFEB B-word; this counter is reset by all these conditions
    if ( fDMB_Header ) { CFEB_SampleWordCount = CFEB_SampleWordCount + 4; }
		
    if (!modeDDUonly) {
      // DCC Header 1 && DCC Header 2
      if( (buf0[3]&0xF000) == 0x5000 && (buf0[0]&0x00FF) == 0x005F &&
	  (buf1[3]&0xFF00) == 0xD900 ){
	if( fDCC_Header ){
	  // == Another DCC Header before encountering DCC Trailer!
	  fERROR[25]=true;
	  bERROR|=0x2000000;
	  fERROR[0]=true;
	  bERROR|=0x1;
	  cerr<<"\n\nDCC Header Occurrence ";
	  cerr<<"  ERROR 25    "<<sERROR[25]<<endl;
	  fDDU_Header = false;

	  // go backward for 3 DDU words ( buf2, buf1, and buf0 )
	  buffer-=12;
	  buf_1 = &(tmpbuf[0]);  // Just for safety
	  buf0  = &(tmpbuf[4]);  // Just for safety
	  buf1  = &(tmpbuf[8]);  // Just for safety
	  buf2  = &(tmpbuf[12]); // Just for safety
	  bzero(tmpbuf,sizeof(unsigned short)*16);
	  return length+12;
	}
      }
    }
    fDCC_Header  = true;
    bzero(fERROR,   sizeof(bool)*nERRORS);
    bzero(fWARNING, sizeof(bool)*nWARNINGS);
    bERROR = 0; bWARNING = 0;
    for(int err=0; err<nERRORS;   ++err) fCHAMB_ERR[err].clear();
    for(int wrn=0; wrn<nWARNINGS; ++wrn) fCHAMB_WRN[wrn].clear();
    bCHAMB_ERR.clear();
    bCHAMB_WRN.clear();

    // == Check for Format Control Words, set proper flags, perform self-consistency checks

    // C-words anywhere besides DDU Header
    if( fDDU_Header && ( (buf0[0]&0xF000)==0xC000 || (buf0[1]&0xF000)==0xC000 || (buf0[2]&0xF000)==0xC000 || (buf0[3]&0xF000)==0xC000 ) &&
	( /*buf_1[0]!=0x8000 ||*/ buf_1[1]!=0x8000 || buf_1[2]!=0x0001 || buf_1[3]!=0x8000 ) ){
      fERROR[0]  = true;
      bERROR    |= 0x1;
      fERROR[20] = true;
      bERROR    |= 0x100000;
      cerr<<"\nDDU Header Occurrence = "<<cntDDU_Headers;
      cerr<<"  ERROR 20 "<<sERROR[20]<<endl;
    }

    // == DDU Header found
    if( /*buf0[0]==0x8000 &&*/ buf0[1]==0x8000 && buf0[2]==0x0001 && buf0[3]==0x8000 ){
      // headerDAV_Active = (buf1[1]<<16) | buf1[0]; // Obsolete since 16.09.05

      if( fDDU_Header ){
	// == Another DDU Header before encountering DDU Trailer!
	fERROR[1]=true;
	bERROR|=0x2;
	fERROR[0] = true;
	bERROR|=0x1;
	cerr<<"\n\nDDU Header Occurrence = "<<cntDDU_Headers;
	cerr<<"  ERROR 1    "<<sERROR[1]<<endl;
	fDDU_Header = false;

	// Part of work for chambers that hasn't been done in absent trailer
	if( fDMB_Header || fDMB_Trailer ){
	  fERROR[5] = true;
	  bERROR   |= 0x20;
	  // Since here there are no chances to know what this chamber was, force it to be -2
	  if( currentChamber == -1 ) currentChamber = -2;
	  fCHAMB_ERR[5].insert(currentChamber);
	  bCHAMB_ERR[currentChamber] |= 0x20;
	  fCHAMB_ERR[0].insert(currentChamber);
	  bCHAMB_ERR[currentChamber] |= 0x1;
	  cerr<<"\n\nDDU Header Occurrence = "<<cntDDU_Headers;
	  cerr<<"  ERROR 5    "<<sERROR[5]<<endl;
	}	// One of DMB Trailers is missing ( or both )
	fDMB_Header  = false;
	fDMB_Trailer = false;

	if( DAV_DMB ){
	  fERROR[24] = true;
	  bERROR    |= 0x1000000;
	}
	DAV_DMB = 0;

	// Unknown chamber denoted as -2
	// If it still remains in any of errors - put it in error 0
	for(int err=1; err<nERRORS; ++err)
	  if( fCHAMB_ERR[err].find(-2) != fCHAMB_ERR[err].end() ){
	    fCHAMB_ERR[0].insert(-2);
	    bCHAMB_ERR[-2] |= 0x1;
	  }

	// go backward for 3 DDU words ( buf2, buf1, and buf0 )
	buffer-=12;
	buf_1 = &(tmpbuf[0]);  // Just for safety
	buf0  = &(tmpbuf[4]);  // Just for safety
	buf1  = &(tmpbuf[8]);  // Just for safety
	buf2  = &(tmpbuf[12]); // Just for safety
	bzero(tmpbuf,sizeof(unsigned short)*16);
	return length+12;
      }

      // Reset all Error and Warning flags to be false
      ///bzero(fERROR,   sizeof(bool)*nERRORS);
      ///bzero(fWARNING, sizeof(bool)*nWARNINGS);
      ///bERROR = 0; bWARNING = 0;
      ///for(int err=0; err<nERRORS;   err++) fCHAMB_ERR[err].clear();
      ///for(int wrn=0; wrn<nWARNINGS; wrn++) fCHAMB_WRN[wrn].clear();
      ///bCHAMB_ERR.clear();
      ///bCHAMB_WRN.clear();
      currentChamber = -1; // Unknown yet

      if( fDDU_Trailer && DDU_WordsSinceLastTrailer != 4 ){
	// == Counted extraneous words between last DDU Trailer and this DDU Header
	fWARNING[0]=true;
	bWARNING|=0x1;
	cerr<<"\nDDU Header Occurrence = "<<cntDDU_Headers;
	cerr<<"  WARNING 0 "<<sWARNING[0]<<" "<<DDU_WordsSinceLastTrailer<<" extra 64-bit words"<<endl;
      }

      fDDU_Header   = true;
      fDDU_Trailer  = false;
      DDU_WordCount = 0;
      fDMB_Header   = false;
      fDMB_Trailer  = false;
      fALCT_Header  = false;
      fTMB_Header   = false;
      uniqueALCT    = true;
      uniqueTMB     = true;
      ALCT_WordsSinceLastHeader = 0;
      ALCT_WordCount            = 0;
      ALCT_WordsExpected        = 0;
      TMB_WordsSinceLastHeader  = 0;
      TMB_WordCount             = 0;
      TMB_WordsExpected         = 0;
      TMB_Tbins                 = 0;
      CFEB_SampleWordCount      = 0;
      CFEB_SampleCount          = 0;
      CFEB_BSampleCount         = 0;

      DAV_DMB = buf1[0]&0xF;

      ++cntDDU_Headers;
      DDU_WordsSinceLastHeader=0; // Reset counter of DDU Words since last DDU Header
      cout<<"\n----------------------------------------------------------"<<endl;
      cout<<"DDU  Header Occurrence "<<cntDDU_Headers<< " L1A = " << ( ((buf_1[2]&0xFFFF) + ((buf_1[3]&0x00FF) << 16)) ) <<endl;
    }


    // == DMB Header found
    if( (buf0[0]&0xF000)==0xA000 && (buf0[1]&0xF000)==0xA000 && (buf0[2]&0xF000)==0xA000 && (buf0[3]&0xF000)==0xA000 ){
      if( fDMB_Header || fDMB_Trailer ){ // F or E  DMB Trailer is missed
	fERROR[5]=true;
	bERROR|=0x20;
	fCHAMB_ERR[5].insert(currentChamber);
	bCHAMB_ERR[currentChamber] |= 0x20;
	fCHAMB_ERR[0].insert(currentChamber);
	bCHAMB_ERR[currentChamber] |= 0x1;
      }
      fDMB_Header  = true;
      fDMB_Trailer = false;

      // If previous DMB record was not assigned to any chamber ( it still has -1 indentificator )
      // let's free -1 identificator for current use and call undefined chamber from previous record -2
      // ( -2 may already exists in this sets but we have nothing to do with it )
      for(int err=0; err<nERRORS; ++err)
	if( fCHAMB_ERR[err].find(-1) != fCHAMB_ERR[err].end() ){
	  fCHAMB_ERR[err].erase(-1);
	  fCHAMB_ERR[err].insert(-2);
	}
      if( fCHAMB_WRN[1].find(-1) != fCHAMB_WRN[1].end() ){
	fCHAMB_WRN[1].erase(-1);
	fCHAMB_WRN[1].insert(-2);
      }
      bCHAMB_ERR[-2] |= bCHAMB_ERR[-1];
      bCHAMB_ERR[-1] = 0;
      bCHAMB_WRN[-2] |= bCHAMB_WRN[-1];
      bCHAMB_WRN[-1] = 0;

      // Chamber id ( DMB_ID + (DMB_CRATE<<4) ) from header
      currentChamber = buf0[1]&0x0FFF;
      ++cntCHAMB_Headers[currentChamber];

      fALCT_Header = false;
      fTMB_Header  = false;
      uniqueALCT   = true;
      uniqueTMB    = true;
      ALCT_WordsSinceLastHeader = 0;
      ALCT_WordCount            = 0;
      ALCT_WordsExpected        = 0;
      TMB_WordsSinceLastHeader  = 0;
      TMB_WordCount             = 0;
      TMB_WordsExpected         = 0;
      TMB_Tbins                 = 0;
      CFEB_SampleWordCount      = 0;
      CFEB_SampleCount          = 0;
      CFEB_BSampleCount         = 0;
      CFEB_CRC                  = 0;

      DAV_DMB--;

      // Print DMB_ID from DMB Header
      cout<<"DMB="<<setw(2)<<setfill('0')<<(buf0[1]&0x000F)<<" ";
      // Print ALCT_DAV and TMB_DAV from DMB Header
      //cout<<setw(1)<<((buf0[0]&0x0020)>>5)<<" "<<((buf0[0]&0x0040)>>6)<<" ";
      cout<<setw(1)<<((buf0[0]&0x0200)>>9)<<" "<<((buf0[0]&0x0800)>>11)<<" "; //change of format 16.09.05
      // Print CFEB_DAV from DMB Header
      cout<<setw(1)<<((buf0[0]&0x0010)>>4)<<((buf0[0]&0x0008)>>3)<<((buf0[0]&0x0004)>>2)<<((buf0[0]&0x0002)>>1)<<(buf0[0]&0x0001);
      // Print DMB Header Tag
      cout << " {";

      // Set variables if we are waiting ALCT, TMB and CFEB records to be present in event
      DAV_ALCT = (buf0[0]&0x0200)>>9;
      DAV_TMB  = (buf0[0]&0x0800)>>11;
      DAV_CFEB = 0;
      if( buf0[0]&0x0001 ) ++DAV_CFEB;
      if( buf0[0]&0x0002 ) ++DAV_CFEB;
      if( buf0[0]&0x0004 ) ++DAV_CFEB;
      if( buf0[0]&0x0008 ) ++DAV_CFEB;
      if( buf0[0]&0x0010 ) ++DAV_CFEB;
    }


    // == ALCT Header found right after DMB Header
    //   (check for all currently reserved/fixed bits in ALCT first 4 words)
    // if( ( (buf0 [0]&0xF800)==0x6000 && (buf0 [1]&0xFF80)==0x0080 && (buf0 [2]&0xF000)==0x0000 && (buf0 [3]&0xc000)==0x0000 )
    if( ( (buf0 [0]&0xF800)==0x6000 && (buf0 [1]&0x8F80)==0x0080 && (buf0 [2]&0x8000)==0x0000 && (buf0 [3]&0xc000)==0x0000 )
	&&
	( (buf_1[0]&0xF000)==0xA000 && (buf_1[1]&0xF000)==0xA000 && (buf_1[2]&0xF000)==0xA000 && (buf_1[3]&0xF000)==0xA000 ) ){
      fALCT_Header              = true;
      ALCT_CRC                  = 0;
      ALCT_WordsSinceLastHeader = 4;

      // Calculate expected number of ALCT words
      if( (buf0[3]&0x0003)==0 ){ ALCT_WordsExpected = 12; }  	// Short Readout

      if( (buf0[1]&0x0003)==1 ){ 					// Full Readout
	ALCT_WordsExpected = ((buf0[1]&0x007c) >> 2) *
	  (	((buf0[3]&0x0001)   )+((buf0[3]&0x0002)>>1)+
		((buf0[3]&0x0004)>>2)+((buf0[3]&0x0008)>>3)+
		((buf0[3]&0x0010)>>4)+((buf0[3]&0x0020)>>5)+
		((buf0[3]&0x0040)>>6) ) * 12 + 12;
      }
      cout<<" <A";
    }

    // == TMB Header found right after DMB Header or right after ALCT Trailer
    if(   (buf0 [0]&0xFFFF)==0x6B0C && (
					( (buf_1[0]&0xF000)==0xA000 && (buf_1[1]&0xF000)==0xA000 && (buf_1[2]&0xF000)==0xA000 && (buf_1[3]&0xF000)==0xA000 )
					||
					( (buf_1[0]&0x0800)==0x0000 && (buf_1[1]&0xF800)==0xD000 && (buf_1[2]&0xFFFF)==0xDE0D && (buf_1[3]&0xF000)==0xD000 )
					// should've been (buf_1[0]&0xF800)==0xD000 - see comments for sERROR[11]
					) )
      {
	//if( (buf_1[2]&0xFFFF)==0xDE0D && (buf_1[3]&0xFC00)!=0xD000 && summer2004 ) ???

	fTMB_Header              = true;
	TMB_CRC                  = 0;
	TMB_WordsSinceLastHeader = 4;

	// Calculate expected number of TMB words (whether RPC included will be known later)
	if ( (buf0[1]&0x3000) == 0x3000) { TMB_WordsExpected = 8; }   // Short Header Only
	if ( (buf0[1]&0x3000) == 0x0000) { TMB_WordsExpected = 32; }  // Long Header Only

	if ( (buf0[1]&0x3000) == 0x1000) {
	  // Full Readout   = 28 + (#Tbins * #CFEBs * 6)
	  TMB_Tbins=(buf0[1]&0x001F);
	  TMB_WordsExpected = 28 + TMB_Tbins * ((buf1[0]&0x00E0)>>5) * 6;
	}

	cout << " <T";
      }

    // == ALCT Trailer found
    if( (buf0[0]&0x0800)==0x0000 && (buf0[1]&0xF800)==0xD000 && (buf0[2]&0xFFFF)==0xDE0D && (buf0[3]&0xF000)==0xD000 ){
      // should've been (buf0[0]&0xF800)==0xD000 - see comments for sERROR[11]

      // Second ALCT -> Lost both previous DMB Trailer and current DMB Header
      if( !uniqueALCT ) currentChamber = -1;
      // Check if this ALCT record have to exist according to DMB Header
      if(   DAV_ALCT  ) DAV_ALCT = false; else DAV_ALCT = true;

      if( !fALCT_Header ){
	fERROR[8] = true;
	bERROR   |= 0x100;
	fCHAMB_ERR[8].insert(currentChamber);
	bCHAMB_ERR[currentChamber] |= 0x100;
	fCHAMB_ERR[0].insert(currentChamber);
	bCHAMB_ERR[currentChamber] |= 0x1;
      } // ALCT Header is missing

      if( (buf0[0]&0xF800)!=0xD000 ){
	fERROR[11] = true;
	bERROR    |= 0x800;
	fCHAMB_ERR[11].insert(currentChamber);
	bCHAMB_ERR[currentChamber] |= 0x800;
	fCHAMB_ERR[0].insert(currentChamber);
	bCHAMB_ERR[currentChamber] |= 0x1;
      } // some bits in 1st D-Trailer are lost

      // Check calculated CRC sum against reported
      if( checkCrcALCT ){
	unsigned long crc = buf0[0] & 0x7ff;
	crc |= ((unsigned long)(buf0[1] & 0x7ff)) << 11;
	if( ALCT_CRC != crc ){
	  fERROR[10] = true;
	  bERROR   |= 0x400;
	  fCHAMB_ERR[10].insert(currentChamber);
	  bCHAMB_ERR[currentChamber] |= 0x400;
	  fCHAMB_ERR[0].insert(currentChamber);
	  bCHAMB_ERR[currentChamber] |= 0x1;
	}
      }

      fALCT_Header = false;
      uniqueALCT   = false;
      CFEB_CRC     = 0;
      //ALCT_WordCount = (buf0[3]&0x03FF);
      ALCT_WordCount = (buf0[3]&0x07FF);
      CFEB_SampleWordCount = 0;
      cout << "A> ";
    }

    // Calculation of CRC sum ( algorithm is written by Madorsky )
    if( fALCT_Header && checkCrcALCT ){
      for(unsigned short j=0, w=0; j<4; ++j){
	w = buf0[j] & 0x7fff;
	for(unsigned long i=15, t=0, ncrc=0; i<16; i--){
	  t = ((w >> i) & 1) ^ ((ALCT_CRC >> 21) & 1);
	  ncrc = (ALCT_CRC << 1) & 0x3ffffc;
	  ncrc |= (t ^ (ALCT_CRC & 1)) << 1;
	  ncrc |= t;
	  ALCT_CRC = ncrc;
	}
      }
    }

    // == Find Correction for TMB_WordsExpected,
    //    should it turn out to be the new RPC-aware format
    if( fTMB_Header && ((buf0[2]&0xFFFF)==0x6E0B) )  {
      TMB_WordsExpectedCorrection =  2 +   // header/trailer for block of RPC raw hits
	//				((buf_1[2]&0x0800)>>11) * ((buf_1[2]&0x0700)>>8) * TMB_Tbins * 2;  // RPC raw hits
	((buf_1[2]&0x0040)>>6) * ((buf_1[2]&0x0030)>>4) * TMB_Tbins * 2;  // RPC raw hits
    }

    // == TMB Trailer found
    if( (buf0[0]&0xF000)==0xD000 && (buf0[1]&0xF000)==0xD000 && (buf0[2]&0xFFFF)==0xDE0F && (buf0[3]&0xF000)==0xD000 ){

      // Second TMB -> Lost both previous DMB Trailer and current DMB Header
      if( !uniqueTMB ) currentChamber = -1;
      // Check if this TMB record have to exist according to DMB Header
      if(   DAV_TMB  ) DAV_TMB = false; else DAV_TMB = true;

      if(!fTMB_Header){
	fERROR[13] = true;
	bERROR    |= 0x2000;
	fCHAMB_ERR[13].insert(currentChamber);
	bCHAMB_ERR[currentChamber] |= 0x2000;
	fCHAMB_ERR[0].insert(currentChamber);
	bCHAMB_ERR[currentChamber] |= 0x1;
      }  // TMB Header is missing

      // Check calculated CRC sum against reported
      if( checkCrcTMB ){
	unsigned long crc = buf0[0] & 0x7ff;
	crc |= ((unsigned long)(buf0[1] & 0x7ff)) << 11;
	if( TMB_CRC != crc ){
	  fERROR[15] = true;
	  bERROR    |= 0x8000;
	  fCHAMB_ERR[15].insert(currentChamber);
	  bCHAMB_ERR[currentChamber] |= 0x8000;
	  fCHAMB_ERR[0].insert(currentChamber);
	  bCHAMB_ERR[currentChamber] |= 0x1;
	}
      }

      fTMB_Header = false;
      uniqueTMB   = false;
      CFEB_CRC     = 0;
      TMB_WordCount = (buf0[3]&0x07FF);

      // == Correct TMB_WordsExpected
      //	1) for 2 optional 0x2AAA and 0x5555 Words in the Trailer
      //    	2) for extra 4 frames in the new RPC-aware format and
      //         for RPC raw hit data, if present
      if( buf_1[1]==0x6E0C ) {
	TMB_WordsExpected = TMB_WordsExpected + 2;	//
	if( buf_1[0]==0x6E04 )
	  TMB_WordsExpected = TMB_WordsExpected + 4 + TMB_WordsExpectedCorrection;
      }

      if( buf_1[3]==0x6E0C && buf_1[2]==0x6E04 )
	TMB_WordsExpected = TMB_WordsExpected + 4 + TMB_WordsExpectedCorrection;

      CFEB_SampleWordCount = 0;
      cout << "T> ";
    }

    if( fTMB_Header && checkCrcTMB ){
      for(unsigned short j=0, w=0; j<4; ++j){
	w = buf0[j] & 0x7fff;
	for(unsigned long i=15, t=0, ncrc=0; i<16; i--){
	  t = ((w >> i) & 1) ^ ((TMB_CRC >> 21) & 1);
	  ncrc = (TMB_CRC << 1) & 0x3ffffc;
	  ncrc |= (t ^ (TMB_CRC & 1)) << 1;
	  ncrc |= t;
	  TMB_CRC = ncrc;
	}
      }
    }


    // == CFEB Sample Trailer found
    if( (buf0[1]&0xF000)==0x7000 && (buf0[2]&0xF000)==0x7000 && (buf0[3]&0xFFFF)==0x7FFF ){
      if((CFEB_SampleCount%8)  == 0   ){ cout<<" <"; }
      if( CFEB_SampleWordCount == 100 ){ cout<<"+";  }
      if( CFEB_SampleWordCount != 100 ){ cout<<"-";
      fERROR[16] = true;
      bERROR    |= 0x10000;
      fCHAMB_ERR[16].insert(currentChamber);
      bCHAMB_ERR[currentChamber] |= 0x10000;
      fCHAMB_ERR[0].insert(currentChamber);
      bCHAMB_ERR[currentChamber] |= 0x1;
      }

      ++CFEB_SampleCount;

      if( (CFEB_SampleCount%8)==0 ){
	cout<<">";
	CFEB_BSampleCount=0;
	// Count CFEBs
	DAV_CFEB--;
      }

      // Check calculated CRC sum against reported
      if( checkCrcCFEB && CFEB_CRC!=buf0[0] ){
	fERROR[18] = true;
	bERROR    |= 0x40000;
	fCHAMB_ERR[18].insert(currentChamber);
	bCHAMB_ERR[currentChamber] |= 0x40000;
	fCHAMB_ERR[0].insert(currentChamber);
	bCHAMB_ERR[currentChamber] |= 0x1;
      }

      CFEB_CRC = 0;
      CFEB_SampleWordCount=0;
    }

    // == If it is nither ALCT record nor TMB - probably it is CFEB record and we try to count CRC sum.
    // It very few words of CFEB occasionaly will be misinterpreted as ALCT or TMB header the result
    // for the CRC sum will be wrong, but other errors of Trailers counting will appear as well
    if( checkCrcCFEB && fDMB_Header && !fTMB_Header && !fALCT_Header && CFEB_SampleWordCount )
      for(int pos=0; pos<4; ++pos)
	CFEB_CRC=(buf0[pos]&0x1fff)^((buf0[pos]&0x1fff)<<1)^(((CFEB_CRC&0x7ffc)>>2)|((0x0003&CFEB_CRC)<<13))^((CFEB_CRC&0x7ffc)>>1);


    // == CFEB B-word found
    if( (buf0[0]&0xF000)==0xB000 && (buf0[1]&0xF000)==0xB000 && (buf0[2]&0xF000)==0xB000 && (buf0[3]&0xF000)==0xB000 ){
      fWARNING[1] = true;
      bWARNING   |= 0x2;
      fCHAMB_WRN[1].insert(currentChamber);
      bCHAMB_WRN[currentChamber] |= 0x2;
      if( (CFEB_SampleCount%8)==0 ){ cout<<" <"; }
      cout<<"B";

      ++CFEB_SampleCount;
      ++CFEB_BSampleCount;

      if( (CFEB_SampleCount%8)==0 ){
	cout << ">";
	CFEB_BSampleCount=0;
      }

      CFEB_SampleWordCount=0;
    }


    // == DMB F-Trailer found
    if( (buf0[0]&0xF000)==0xF000 && (buf0[1]&0xF000)==0xF000 && (buf0[2]&0xF000)==0xF000 && (buf0[3]&0xF000)==0xF000 ){
      if(!fDMB_Header){
	fERROR[6] = true;
	bERROR   |= 0x40;
	fCHAMB_ERR[6].insert(currentChamber);
	bCHAMB_ERR[currentChamber] |= 0x40;
	DAV_DMB--;
      }	// DMB Header is missing
      fDMB_Header  = false;
      fDMB_Trailer = true;
      uniqueALCT   = true;
      uniqueTMB    = true;

      // Finally check if DAVs were correct
      if( DAV_ALCT ){
	fERROR[21] = true;
	bERROR   |= 0x200000;
	fCHAMB_ERR[21].insert(currentChamber);
	bCHAMB_ERR[currentChamber] |= 0x200000;
      }
      if( DAV_TMB  ){
	fERROR[22] = true;
	bERROR   |= 0x400000;
	fCHAMB_ERR[22].insert(currentChamber);
	bCHAMB_ERR[currentChamber] |= 0x400000;
      }
      if( DAV_CFEB && DAV_CFEB!=-16 ){
	fERROR[23] = true;
	bERROR   |= 0x800000;
	fCHAMB_ERR[23].insert(currentChamber);
	bCHAMB_ERR[currentChamber] |= 0x800000;
      }

      // If F-Trailer is lost then do necessary work here
      if( (buf1[0]&0xF000)!=0xE000 || (buf1[1]&0xF000)!=0xE000 || (buf1[2]&0xF000)!=0xE000 || (buf1[3]&0xF000)!=0xE000 ){
	for(int err=1; err<nERRORS; ++err)
	  if( fCHAMB_ERR[err].find(currentChamber) != fCHAMB_ERR[err].end() ){
	    fCHAMB_ERR[0].insert(currentChamber);
	    bCHAMB_ERR[currentChamber] |= 0x1;
	  }
	// Reset chamber id
	currentChamber=-1;
	/*
	  for(int err=0; err<nERRORS; err++)
	  if( fCHAMB_ERR[err].find(-1) != fCHAMB_ERR[err].end() )
	  fCHAMB_ERR[err].erase(-1);
	  if( fCHAMB_WRN[1].find(-1) != fCHAMB_WRN[1].end() )
	  fCHAMB_WRN[1].erase(-1);
	  bCHAMB_ERR[-1] = 0;
	  bCHAMB_WRN[-1] = 0;
	*/
      }

      // Print DMB F-Trailer marker
      cout << " }";
    }

    // == DMB E-Trailer found
    if( (buf0[0]&0xF000)==0xE000 && (buf0[1]&0xF000)==0xE000 && (buf0[2]&0xF000)==0xE000 && (buf0[3]&0xF000)==0xE000 ){
      if( !fDMB_Header && !fDMB_Trailer ) DAV_DMB--; // both DMB Header and DMB F-Trailer were missing

      fDMB_Header  = false;

      // If chamber id is unknown it is time to find it out
      if( currentChamber==-1 ){
	currentChamber = buf0[1]&0x0FFF;
	for(int err=0; err<nERRORS; ++err)
	  if( fCHAMB_ERR[err].find(-1) != fCHAMB_ERR[err].end() ){
	    fCHAMB_ERR[err].insert(currentChamber);
	    fCHAMB_ERR[err].erase(-1);
	  }
	if( fCHAMB_WRN[1].find(-1) != fCHAMB_WRN[1].end() ){
	  fCHAMB_WRN[1].insert(currentChamber);
	  fCHAMB_WRN[1].erase(-1);
	}
	bCHAMB_ERR[currentChamber] = bCHAMB_ERR[-1];
	bCHAMB_ERR[-1] = 0;
	bCHAMB_WRN[currentChamber] = bCHAMB_WRN[-1];
	bCHAMB_WRN[-1] = 0;
      }
      ++cntCHAMB_Trailers[buf0[1]&0x0FFF];

      // Lost DMB F-Trailer before
      if( !fDMB_Trailer ){
	fERROR[6] = true;
	bERROR   |= 0x40;
	fCHAMB_ERR[6].insert(currentChamber);
	bCHAMB_ERR[currentChamber] |= 0x40;
	fCHAMB_ERR[0].insert(currentChamber);
	bCHAMB_ERR[currentChamber] |= 0x1;
	// Check if DAVs were correct here
	if( DAV_ALCT ){
	  fERROR[21] = true;
	  bERROR   |= 0x200000;
	  fCHAMB_ERR[21].insert(currentChamber);
	  bCHAMB_ERR[currentChamber] |= 0x200000;
	}
	if( DAV_TMB  ){
	  fERROR[22] = true;
	  bERROR   |= 0x400000;
	  fCHAMB_ERR[22].insert(currentChamber);
	  bCHAMB_ERR[currentChamber] |= 0x400000;
	}
	if( DAV_CFEB && DAV_CFEB!=-16){
	  fERROR[23] = true;
	  bERROR   |= 0x800000;
	  fCHAMB_ERR[23].insert(currentChamber);
	  bCHAMB_ERR[currentChamber] |= 0x800000;
	}
      }
      fDMB_Trailer = false;

      if( !fALCT_Header && ( ALCT_WordsSinceLastHeader!=ALCT_WordCount || ALCT_WordsSinceLastHeader!=ALCT_WordsExpected ) ){
	fERROR[9] = true;
	bERROR   |= 0x200;
	fCHAMB_ERR[9].insert(currentChamber);
	bCHAMB_ERR[currentChamber] |= 0x200;
      } // ALCT Word Count Error

      if( !fTMB_Header && ( TMB_WordsSinceLastHeader!=TMB_WordCount || TMB_WordsSinceLastHeader!=TMB_WordsExpected ) ){
	fERROR[14] = true;
	bERROR    |= 0x4000;
	fCHAMB_ERR[14].insert(currentChamber);
	bCHAMB_ERR[currentChamber] |= 0x4000;
      } // TMB Word Count Error

      if( (CFEB_SampleCount%8)!=0 ){
	fERROR[17] = true;
	bERROR    |= 0x20000;
	fCHAMB_ERR[17].insert(currentChamber);
	bCHAMB_ERR[currentChamber] |= 0x20000;
      } // Number of CFEB samples != 8*n

      if(fALCT_Header) {
	fERROR[7] = true;  // ALCT Trailer is missing
	bERROR   |= 0x80;
	fCHAMB_ERR[7].insert(currentChamber);
	bCHAMB_ERR[currentChamber] |= 0x80;
	ALCT_WordsSinceLastHeader = 0;
	ALCT_WordsExpected        = 0;
      }

      if(fTMB_Header) {
	fERROR[12]=true;	// TMB Trailer is missing
	bERROR   |= 0x1000;
	fCHAMB_ERR[12].insert(currentChamber);
	bCHAMB_ERR[currentChamber] |= 0x1000;
	TMB_WordsSinceLastHeader = 0;
	TMB_WordsExpected        = 0;
      }

      //
      for(int err=0; err<nERRORS; ++err)
	if( fCHAMB_ERR[err].find(-1) != fCHAMB_ERR[err].end() ){
	  fCHAMB_ERR[err].erase(-1);
	  fCHAMB_ERR[err].insert(-2);
	}
      if( fCHAMB_WRN[1].find(-1) != fCHAMB_WRN[1].end() ){
	fCHAMB_WRN[1].erase(-1);
	fCHAMB_WRN[1].insert(-2);
      }
      bCHAMB_ERR[-2] |= bCHAMB_ERR[-1];
      bCHAMB_ERR[-1] = 0;
      bCHAMB_WRN[-2] |= bCHAMB_WRN[-1];
      bCHAMB_WRN[-1] = 0;

      if( currentChamber != -1 )
	for(int err=1; err<nERRORS; ++err)
	  if( fCHAMB_ERR[err].find(currentChamber) != fCHAMB_ERR[err].end() ){
	    fCHAMB_ERR[0].insert(currentChamber);
	    bCHAMB_ERR[currentChamber] |= 0x1;
	  }

      currentChamber=-1;

      // Print DMB E-Trailer marker
      cout<<" DMB="<<(buf0[1]&0x000F);
      cout << "; "
	   << ALCT_WordsSinceLastHeader << "-"
	   << ALCT_WordCount << "-"
	   << ALCT_WordsExpected
	   << "      "
	   << TMB_WordsSinceLastHeader << "-"
	   << TMB_WordCount << "-"
	   << TMB_WordsExpected
	   << endl;
    }


    // == DDU Trailer found
    if( buf0[0]==0x8000 && buf0[1]==0x8000 && buf0[2]==0xFFFF && buf0[3]==0x8000 ){
      // Obsolete since 16.09.05
      //			if( headerDAV_Active != ((buf1[1]<<16) | buf1[0]) ){
      //				fERROR[0]  = true;
      //				fERROR[21] = true;
      //				bERROR|=0x200000;
      //				//cerr<<"  ERROR 21   "<<sERROR[21]<<endl;
      //			}
      //			headerDAV_Active = -1;

      if(fDDU_Trailer){
	fERROR[2] = true;
	bERROR   |= 0x4;
      } // DDU Header is missing
      fDDU_Trailer=true;
      fDDU_Header=false;

      if( fDMB_Header || fDMB_Trailer ){
	fERROR[5] = true;
	bERROR   |= 0x20;
	fCHAMB_ERR[5].insert(currentChamber);
	bCHAMB_ERR[currentChamber] |= 0x20;
	fCHAMB_ERR[0].insert(currentChamber);
	bCHAMB_ERR[currentChamber] |= 0x20;
      }	// DMB Trailer is missing
      fDMB_Header  = false;
      fDMB_Trailer = false;

      currentChamber=-1;

      for(int err=0; err<nERRORS; ++err)
	if( fCHAMB_ERR[err].find(-1) != fCHAMB_ERR[err].end() ){
	  fCHAMB_ERR[err].erase(-1);
	  fCHAMB_ERR[err].insert(-2);
	}
      if( fCHAMB_WRN[1].find(-1) != fCHAMB_WRN[1].end() ){
	fCHAMB_WRN[1].erase(-1);
	fCHAMB_WRN[1].insert(-2);
      }
      bCHAMB_ERR[-2] |= bCHAMB_ERR[-1];
      bCHAMB_ERR[-1] = 0;
      bCHAMB_WRN[-2] |= bCHAMB_WRN[-1];
      bCHAMB_WRN[-1] = 0;

      for(int err=1; err<nERRORS; ++err)
	if( fCHAMB_ERR[err].find(-2) != fCHAMB_ERR[err].end() ){
	  fCHAMB_ERR[0].insert(-2);
	  bCHAMB_ERR[-2] |= 0x1;
	}


      ++cntDDU_Trailers; // Increment DDUTrailer counter

      // == Combining 2 words into 24bit value
      DDU_WordCount = buf2[2] | ((buf2[3] & 0xFF) <<16) ;

      if( (DDU_WordsSinceLastHeader+4) != DDU_WordCount ){
	fERROR[4] = true;
	bERROR   |= 0x10;
      }

      if( DAV_DMB ){
	fERROR[24] = true;
	bERROR    |= 0x1000000;
      }

      cout<<"DDU Trailer Occurrence "<<cntDDU_Trailers<<endl;
      cout<<"----------------------------------------------------------"<<endl;
      cout<<"DDU 64-bit words = Actual - DDUcounted ="<<DDU_WordsSinceLastHeader+4<<"-"<<DDU_WordCount<<endl;

      // increment statistics Errors and Warnings (i=0 case is handled in DDU Header)
      for(int err=1; err<nERRORS; ++err){
	if( fERROR[err] ){
	  fERROR[0] = true;
	  bERROR |= 0x1;
	  cerr<<"\nDDU Header Occurrence = "<<cntDDU_Headers;
	  cerr<<"  ERROR "<<err<<"  " <<sERROR[err]<<endl;
	}
      }
      for(int wrn=1; wrn<nWARNINGS; ++wrn){
	if( fWARNING[wrn] ){
	  cout<<"\nDDU Header Occurrence = "<<cntDDU_Headers;
	  cout<<"  WARNING "<<wrn<<"  "<<sWARNING[wrn]<<endl;
	}
      }

      DDU_WordsSinceLastHeader=0;
      DDU_WordsSinceLastTrailer=0;
      if (modeDDUonly) {
	buffer+=4;
	buf_1 = &(tmpbuf[0]);  // Just for safety
	buf0  = &(tmpbuf[4]);  // Just for safety
	buf1  = &(tmpbuf[8]);  // Just for safety
	buf2  = &(tmpbuf[12]); // Just for safety
	bzero(tmpbuf, sizeof(short)*16);
	return length-4;
      }
    }

    if (!modeDDUonly) {
      // DCC Trailer 1 && DCC Trailer 2
      if( (buf1[3]&0xFF00) == 0xEF00 &&
	  (buf2[3]&0xFF00) == 0xAF00 ){
	if(fDCC_Trailer){
	  fERROR[26] = true;
	  bERROR|=0x4000000;
	} // DCC Header is missing
	fDCC_Trailer=true;
	fDCC_Header=false;

	if( fDDU_Header ){
	  // == DDU Trailer is missing
	  fERROR[1]=true;
	  bERROR|=0x2;
	  fERROR[0] = true;
	  bERROR|=0x1;
	}

	buffer+=4;
	buf_1 = &(tmpbuf[0]);  // Just for safety
	buf0  = &(tmpbuf[4]);  // Just for safety
	buf1  = &(tmpbuf[8]);  // Just for safety
	buf2  = &(tmpbuf[12]); // Just for safety
	bzero(tmpbuf, sizeof(short)*16);
	return length-4;
      }
    }

    length-=4;
    buffer+=4;
  }
  //Store the tail of the buffer
  buf_1 = &(tmpbuf[0]);
  buf0  = &(tmpbuf[4]);
  buf1  = &(tmpbuf[8]);
  buf2  = &(tmpbuf[12]);
  memcpy((void*)tmpbuf,(void*)(buffer-16),sizeof(short)*16);

  return -1;
}

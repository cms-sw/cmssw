#include "EventFilter/CSCRawToDigi/interface/CSCDCCExaminer.h"

#ifdef LOCAL_UNPACK
#include <string.h>

#else

#include <cstring>
#include <cassert>

/*
#ifdef CSC_DEBUG
#include <iostream>
#define COUT std::COUT
#define CERR std::CERR
#else
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#define COUT LogTrace("CSCDCCExaminer")
// #define CERR edm::LogWarning("CSCDCCExaminer")
#define CERR LogDebug("CSCDCCExaminer")
#endif
*/

#endif

#include <iomanip>
using namespace std;

void CSCDCCExaminer::crcALCT(bool enable)
{
  checkCrcALCT = enable;
  if( checkCrcALCT )
    sERROR[10] = "ALCT CRC Error                                   ";
  else
    sERROR[10] = "ALCT CRC Error ( disabled )                      ";
}

void CSCDCCExaminer::crcTMB(bool enable)
{
  checkCrcTMB = enable;
  if( checkCrcTMB )
    sERROR[15] = "TMB CRC Error                                    ";
  else
    sERROR[15] = "TMB CRC Error ( disabled )                       ";
}

void CSCDCCExaminer::crcCFEB(bool enable)
{
  checkCrcCFEB = enable;
  if( checkCrcCFEB )
    sERROR[18] = "CFEB CRC Error                                   ";
  else
    sERROR[18] = "CFEB CRC Error ( disabled )                      ";
}

void CSCDCCExaminer::modeDDU(bool enable)
{
  modeDDUonly = enable;
  if( modeDDUonly)
    {
      sERROR[25] = "DCC Trailer Missing                              ";
      sERROR[26] = "DCC Header Missing                               ";
    }
  else
    {
      sERROR[25] = "DCC Trailer Missing (disabled)                   ";
      sERROR[26] = "DCC Header Missing (disabled)                    ";
    }

}


CSCDCCExaminer::CSCDCCExaminer(ExaminerMaskType mask)
  :nERRORS(29),nWARNINGS(5),nPAYLOADS(16),nSTATUSES(29),sERROR(nERRORS),sWARNING(nWARNINGS),sERROR_(nERRORS),sWARNING_(nWARNINGS),sDMBExpectedPayload(nPAYLOADS),sDMBEventStaus(nSTATUSES),examinerMask(mask)
{

#ifdef LOCAL_UNPACK
  COUT.redirect(std::cout);
  CERR.redirect(std::cerr);
#endif

  sERROR[0] = " Any errors                                       ";
  sERROR[1] = " DDU Trailer Missing                              ";
  sERROR[2] = " DDU Header Missing                               ";
  sERROR[3] = " DDU CRC Error (not yet implemented)              ";
  sERROR[4] = " DDU Word Count Error                             ";
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
  sERROR[27] = "DMB DAV vs. DMB Active mismatch Error            ";
  sERROR[28] = "Extra words between DDU Header and first DMB header";

  //	sERROR[21] = "DDU Header vs. Trailer mismatch for DAV or Avtive"; // oboslete since 16.09.05

  sWARNING[0] = " Extra words between DDU Trailer and DDU Header ";
  sWARNING[1] = " DDU Header Incomplete                          ";

  sDMBExpectedPayload[0]  = "CFEB1_ACTIVE";
  sDMBExpectedPayload[1]  = "CFEB2_ACTIVE";
  sDMBExpectedPayload[2]  = "CFEB3_ACTIVE";
  sDMBExpectedPayload[3]  = "CFEB4_ACTIVE";
  sDMBExpectedPayload[4]  = "CFEB5_ACTIVE";
  sDMBExpectedPayload[5]  = "ALCT_DAV";
  sDMBExpectedPayload[6]  = "TMB_DAV";
  sDMBExpectedPayload[7]  = "CFEB1_DAV";
  sDMBExpectedPayload[8]  = "CFEB2_DAV";
  sDMBExpectedPayload[9]  = "CFEB3_DAV";
  sDMBExpectedPayload[10] = "CFEB4_DAV";
  sDMBExpectedPayload[11] = "CFEB5_DAV";
  /// 2013 Format additions
  sDMBExpectedPayload[12] = "CFEB6_DAV";
  sDMBExpectedPayload[13] = "CFEB7_DAV";
  sDMBExpectedPayload[14]  = "CFEB6_ACTIVE";
  sDMBExpectedPayload[15]  = "CFEB7_ACTIVE";


  sDMBEventStaus[0]  = "ALCT_FIFO_FULL";
  sDMBEventStaus[1]  = "TMB_FIFO_FULL";
  sDMBEventStaus[2]  = "CFEB1_FIFO_FULL";
  sDMBEventStaus[3]  = "CFEB2_FIFO_FULL";
  sDMBEventStaus[4]  = "CFEB3_FIFO_FULL";
  sDMBEventStaus[5]  = "CFEB4_FIFO_FULL";
  sDMBEventStaus[6]  = "CFEB5_FIFO_FULL";
  sDMBEventStaus[7]  = "ALCT_START_TIMEOUT";
  sDMBEventStaus[8]  = "TMB_START_TIMEOUT";
  sDMBEventStaus[9]  = "CFEB1_START_TIMEOUT";
  sDMBEventStaus[10] = "CFEB2_START_TIMEOUT";
  sDMBEventStaus[11] = "CFEB3_START_TIMEOUT";
  sDMBEventStaus[12] = "CFEB4_START_TIMEOUT";
  sDMBEventStaus[13] = "CFEB5_START_TIMEOUT";
  sDMBEventStaus[14] = "ALCT_END_TIMEOUT";
  sDMBEventStaus[15] = "TMB_END_TIMEOUT";
  sDMBEventStaus[16] = "CFEB1_END_TIMEOUT";
  sDMBEventStaus[17] = "CFEB2_END_TIMEOUT";
  sDMBEventStaus[18] = "CFEB3_END_TIMEOUT";
  sDMBEventStaus[19] = "CFEB4_END_TIMEOUT";
  sDMBEventStaus[20] = "CFEB5_END_TIMEOUT";
  sDMBEventStaus[21] = "CFEB Active-DAV mismatch";
  sDMBEventStaus[22] = "B-words found";
  /// 2013 Format additions
  sDMBEventStaus[23] = "CFEB6_FIFO_FULL";
  sDMBEventStaus[24] = "CFEB7_FIFO_FULL";
  sDMBEventStaus[25] = "CFEB6_START_TIMEOUT";
  sDMBEventStaus[26] = "CFEB7_START_TIMEOUT";
  sDMBEventStaus[27] = "CFEB6_END_TIMEOUT";
  sDMBEventStaus[28] = "CFEB7_END_TIMEOUT";



  sERROR_[0] = " Any errors: 00";
  sERROR_[1] = " DDU Trailer Missing: 01";
  sERROR_[2] = " DDU Header Missing: 02";
  sERROR_[3] = " DDU CRC Error (not yet implemented): 03";
  sERROR_[4] = " DDU Word Count Error: 04";
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
  sERROR_[27] = "DMB DAV vs. DMB Active mismatch Error: 27";
  sERROR_[28] = "Extra words between DDU Header and first DMB header: 28";
  //	sERROR_[21] = "DDU Header vs. Trailer mismatch for DAV or Avtive: 21"; // oboslete since 16.09.05

  sWARNING_[0] = " Extra words between DDU Trailer and DDU Header: 00";
  sWARNING_[1] = " DDU Header Incomplete: 02";

  fDCC_Header  = false;
  fDCC_Trailer = false;
  fDDU_Header  = false;
  fDDU_Trailer = false;
  fDMB_Header  = false;
  fDMB_Trailer = false;
  fALCT_Header = false;
  fTMB_Header  = false;
  fALCT_Format2007 = true;
  fTMB_Format2007  = true;
  fFormat2013 = false;

  cntDDU_Headers  = 0;
  cntDDU_Trailers = 0;
  cntCHAMB_Headers.clear();
  cntCHAMB_Trailers.clear();

  DAV_ALCT = false;
  DAV_TMB  = false;
  DAV_CFEB = 0;
  DMB_Active  = 0;
  nDMBs = 0;
  DDU_WordsSinceLastHeader     = 0;
  DDU_WordCount                = 0;
  DDU_WordMismatch_Occurrences = 0;
  DDU_WordsSinceLastTrailer    = 0;
  ALCT_ZSE                     = 0;
  nWG_round_up                 = 0;

  TMB_WordsRPC  = 0;
  TMB_Firmware_Revision = 0;
  DDU_Firmware_Revision = 0;
  zeroCounts();

  checkCrcALCT = false;
  ALCT_CRC=0;
  checkCrcTMB  = false;
  TMB_CRC=0;
  checkCrcCFEB = false;
  CFEB_CRC=0;

  modeDDUonly = false;
  sourceID    = 0xFFF;
  currentChamber = -1;

  //headerDAV_Active = -1; // Trailer vs. Header check // Obsolete since 16.09.05

  clear();
  buf_1 = &(tmpbuf[0]);
  buf0  = &(tmpbuf[4]);
  buf1  = &(tmpbuf[8]);
  buf2  = &(tmpbuf[12]);

  bzero(tmpbuf, sizeof(uint16_t)*16);
}

int32_t CSCDCCExaminer::check(const uint16_t* &buffer, int32_t length)
{
  if( length<=0 ) return -1;

  /// 'buffer' is a sliding pointer; keep track of the true buffer
  buffer_start = buffer;


  /// Check for presence of data blocks inside TMB data
  bool fTMB_Scope_Start = false;
  bool fTMB_MiniScope_Start = false;
  bool fTMB_RPC_Start = false;
  bool fTMB_BlockedCFEBs_Start = false;

  bool fTMB_Scope = false;
  bool fTMB_MiniScope = false;
  bool fTMB_RPC = false;
  bool fTMB_BlockedCFEBs = false;

  fTMB_Scope = fTMB_Scope && true; // WARNING in 5_0_X

  while( length>0 )
    {
      // == Store last 4 read buffers in pipeline-like memory (note that memcpy works quite slower!)
      buf_2 = buf_1;         //  This bufer was not needed so far
      buf_1 = buf0;
      buf0  = buf1;
      buf1  = buf2;
      buf2  = buffer;

      // check for too long event
      if(!fERROR[19] && DDU_WordsSinceLastHeader>100000 )
        {
          fERROR[19] = true;
          bERROR    |= 0x80000;
        }

      // increment counter of 64-bit words since last DDU Header
      // this counter is reset if DDU Header is found
      if ( fDDU_Header )
        {
          ++DDU_WordsSinceLastHeader;
        }

      // increment counter of 64-bit words since last DDU Trailer
      // this counter is reset if DDU Trailer is found
      if ( fDDU_Trailer )
        {
          ++DDU_WordsSinceLastTrailer;
        }

      /// increment counter of 16-bit words since last DMB*ALCT Header match
      /// this counter is reset if ALCT Header is found right after DMB Header
      if ( fALCT_Header )
        {
          /// decode the actual counting if zero suppression enabled
          if(ALCT_ZSE)
            {
              for(int g=0; g<4; g++)
                {
                  if(buf0[g]==0x1000)
                    {
                      ALCT_WordsSinceLastHeader = ALCT_WordsSinceLastHeader + nWG_round_up;
                    }
                  else if(buf0[g]!=0x3000) ALCT_WordsSinceLastHeader = ALCT_WordsSinceLastHeader + 1;
                }
            }
          else ALCT_WordsSinceLastHeader = ALCT_WordsSinceLastHeader + 4;
          /// increment counter of 16-bit words without zero suppression decoding
          ALCT_WordsSinceLastHeaderZeroSuppressed = ALCT_WordsSinceLastHeaderZeroSuppressed + 4;
        }

      // increment counter of 16-bit words since last DMB*TMB Header match
      // this counter is reset if TMB Header is found right after DMB Header or ALCT Trailer
      if ( fTMB_Header )
        {
          TMB_WordsSinceLastHeader = TMB_WordsSinceLastHeader + 4;
        }

      // increment counter of 16-bit words since last of DMB Header, ALCT Trailer, TMB Trailer,
      // CFEB Sample Trailer, CFEB B-word; this counter is reset by all these conditions
      if ( fDMB_Header )
        {
          CFEB_SampleWordCount = CFEB_SampleWordCount + 4;
        }

      // If DDU header is missing we set unphysical 0xFFF value for DDU id
      if( !fDDU_Header )
        {
          sourceID=0xFFF;
        }


      if (!modeDDUonly)
        {
          // DCC Header 1 && DCC Header 2
          // =VB= Added support for Sep. 2008 CMS DAQ DCC format
          if ( ( ( (buf0[3]&0xF000) == 0x5000 && (buf0[0]&0x00FF) == 0x005F )
                 ||
                 ( (buf0[3]&0xF000) == 0x5000 && (buf0[0]&0x000F) == 0x0008 ) )
               &&
               // =VB= Why 0xD900 signature word if only 0xD part is constant???
               // (buf1[3]&0xFF00) == 0xD900 )
               (buf1[3]&0xF000) == 0xD000 )
            {
              if( fDCC_Header )
                {
                  // == Another DCC Header before encountering DCC Trailer!
                  fERROR[25]=true;
                  bERROR|=0x2000000;
                  fERROR[0]=true;
                  bERROR|=0x1;
#ifdef LOCAL_UNPACK
                  CERR<<"\n\nDCC Header Occurrence ";
                  CERR<<"  ERROR 25    "<<sERROR[25]<<endl;
#endif
                  fDDU_Header = false;

                  // go backward for 3 DDU words ( buf2, buf1, and buf0 )
                  buffer-=12;
                  buf_1 = &(tmpbuf[0]);  // Just for safety
                  buf0  = &(tmpbuf[4]);  // Just for safety
                  buf1  = &(tmpbuf[8]);  // Just for safety
                  buf2  = &(tmpbuf[12]); // Just for safety
                  bzero(tmpbuf,sizeof(uint16_t)*16);
                  sync_stats();
                  return length+12;
                }

              fDCC_Header  = true;
              clear();
            }
        }
      // == Check for Format Control Words, set proper flags, perform self-consistency checks

      // C-words anywhere besides DDU Header
      if( fDDU_Header && ( (buf0[0]&0xF000)==0xC000 || (buf0[1]&0xF000)==0xC000 || (buf0[2]&0xF000)==0xC000 || (buf0[3]&0xF000)==0xC000 ) &&
          ( /*buf_1[0]!=0x8000 ||*/ buf_1[1]!=0x8000 || buf_1[2]!=0x0001 || buf_1[3]!=0x8000 ) )
        {
          fERROR[0]  = true;
          bERROR    |= 0x1;
          fERROR[20] = true;
          bERROR    |= 0x100000;
          // fCHAMB_ERR[20].insert(currentChamber);
          // bCHAMB_ERR[currentChamber] |= 0x100000;
#ifdef LOCAL_UNPACK
          CERR<<"\nDDU Header Occurrence = "<<cntDDU_Headers;
          CERR<<"  ERROR 20 "<<sERROR[20]<<endl;
#endif
        }

      // == DDU Header found
      if( /*buf0[0]==0x8000 &&*/ buf0[1]==0x8000 && buf0[2]==0x0001 && buf0[3]==0x8000 )
        {
          // headerDAV_Active = (buf1[1]<<16) | buf1[0]; // Obsolete since 16.09.05
/////////////////////////////////////////////////////////
          checkDAVs();
          checkTriggerHeadersAndTrailers();
/////////////////////////////////////////////////////////

          if( fDDU_Header )
            {
              // == Another DDU Header before encountering DDU Trailer!
              fERROR[1]=true;
              bERROR|=0x2;
              fERROR[0] = true;
              bERROR|=0x1;
#ifdef LOCAL_UNPACK
              CERR<<"\n\nDDU Header Occurrence = "<<cntDDU_Headers;
              CERR<<"  ERROR 1    "<<sERROR[1]<<endl;
#endif
              fDDU_Header = false;

              // Part of work for chambers that hasn't been done in absent trailer
              if( fDMB_Header || fDMB_Trailer )
                {
                  fERROR[5] = true;
                  bERROR   |= 0x20;
                  // Since here there are no chances to know what this chamber was, force it to be -2
                  if( currentChamber == -1 ) currentChamber = -2;
                  fCHAMB_ERR[5].insert(currentChamber);
                  bCHAMB_ERR[currentChamber] |= 0x20;
                  fCHAMB_ERR[0].insert(currentChamber);
                  bCHAMB_ERR[currentChamber] |= 0x1;
#ifdef LOCAL_UNPACK
                  CERR<<"\n\nDDU Header Occurrence = "<<cntDDU_Headers;
                  CERR<<"  ERROR 5    "<<sERROR[5]<<endl;
#endif
                }	// One of DMB Trailers is missing ( or both )
              fDMB_Header  = false;
              fDMB_Trailer = false;

              if( DMB_Active!=nDMBs )
                {
                  fERROR[24] = true;
                  bERROR    |= 0x1000000;
                }
              DMB_Active = 0;
              nDMBs = 0;

              // Unknown chamber denoted as -2
              // If it still remains in any of errors - put it in error 0
              for(int err=1; err<nERRORS; ++err)
                if( fCHAMB_ERR[err].find(-2) != fCHAMB_ERR[err].end() )
                  {
                    fCHAMB_ERR[0].insert(-2);
                    bCHAMB_ERR[-2] |= 0x1;
                  }

              bDDU_ERR[sourceID] |= bERROR;
              bDDU_WRN[sourceID] |= bWARNING;

              // go backward for 3 DDU words ( buf2, buf1, and buf0 )
              buffer-=12;
              buf_1 = &(tmpbuf[0]);  // Just for safety
              buf0  = &(tmpbuf[4]);  // Just for safety
              buf1  = &(tmpbuf[8]);  // Just for safety
              buf2  = &(tmpbuf[12]); // Just for safety
              bzero(tmpbuf,sizeof(uint16_t)*16);
              sync_stats();
              return length+12;
            }


          currentChamber = -1; // Unknown yet

          if( fDDU_Trailer && DDU_WordsSinceLastTrailer != 4 )
            {
              // == Counted extraneous words between last DDU Trailer and this DDU Header
              fWARNING[0]=true;
              bWARNING|=0x1;
#ifdef LOCAL_UNPACK
              CERR<<"\nDDU Header Occurrence = "<<cntDDU_Headers;
              CERR<<"  WARNING 0 "<<sWARNING[0]<<" "<<DDU_WordsSinceLastTrailer<<" extra 64-bit words"<<endl;
#endif
            }

          sourceID      = ((buf_1[1]&0xF)<<8) | ((buf_1[0]&0xFF00)>>8);

          ///* 2013 Data format version check
          DDU_Firmware_Revision    = (buf_1[0] >> 4) & 0xF;
          if (DDU_Firmware_Revision > 6)
            {
              fFormat2013 = true;
              modeDDUonly = true; // =VB= Force to use DDU only mode (no DCC Data)
            }

          fDDU_Header   = true;
          fDDU_Trailer  = false;
          DDU_WordCount = 0;
          fDMB_Header   = false;
          fDMB_Trailer  = false;
          fALCT_Header  = false;
          fALCT_Format2007= true;
          fTMB_Header   = false;
          fTMB_Format2007= true;
          uniqueALCT    = true;
          uniqueTMB     = true;
          zeroCounts();

          if (modeDDUonly)
            {
              fDCC_Header  = true;
              clear();
            }

          dduBuffers[sourceID] = buf_1;
          dduOffsets[sourceID] = buf_1-buffer_start;
          dduSize   [sourceID] = 0;
          dmbBuffers[sourceID].clear();
          dmbOffsets[sourceID].clear();
          dmbSize   [sourceID].clear();

          // Reset all Error and Warning flags to be false
          bDDU_ERR[sourceID] = 0;
          bDDU_WRN[sourceID] = 0;
          bERROR             = 0;
          bWARNING           = 0;
          bzero(fERROR,   sizeof(bool)*nERRORS);
          bzero(fWARNING, sizeof(bool)*nWARNINGS);

          nDMBs      = 0;
          DMB_Active = buf1[0]&0xF;
          DAV_DMB    = buf1[1]&0x7FFF;

          int nDAV_DMBs=0;
          for(int bit=0; bit<15; bit++) if( DAV_DMB&(1<<bit) ) nDAV_DMBs++;
          if(DMB_Active!=nDAV_DMBs)
            {
              fERROR[27] = true;
              bERROR    |= 0x8000000;
            }

          if( (buf_1[3]&0xF000)!=0x5000 )
            {
              fWARNING[1]=true;
              bWARNING|=0x2;
#ifdef LOCAL_UNPACK
              CERR<<"\nDDU Header Occurrence = "<<cntDDU_Headers;
              CERR<<"  WARNING 1 "<<sWARNING[1]<<". What must have been Header 1: 0x"<<std::hex<<buf_1[0]<<" 0x"<<buf_1[1]<<" 0x"<<buf_1[2]<<" 0x"<<buf_1[3]<<std::dec<<endl;
#endif
            }

          ++cntDDU_Headers;
          DDU_WordsSinceLastHeader=0; // Reset counter of DDU Words since last DDU Header
#ifdef LOCAL_UNPACK
          COUT<<"\n----------------------------------------------------------"<<endl;
          COUT<<"DDU  Header Occurrence "<<cntDDU_Headers<< " L1A = " << ( ((buf_1[2]&0xFFFF) + ((buf_1[3]&0x00FF) << 16)) ) <<endl;
#endif

        }

      // == DMB Header found
      if( (buf0[0]&0xF000)==0xA000 && (buf0[1]&0xF000)==0xA000 && (buf0[2]&0xF000)==0xA000 && (buf0[3]&0xF000)==0xA000 )
        {
/////////////////////////////////////////////////////////
          checkDAVs();
          checkTriggerHeadersAndTrailers();
/////////////////////////////////////////////////////////

          if( DDU_WordsSinceLastHeader>3 && !fDMB_Header && !fDMB_Trailer && !nDMBs )
            {
              fERROR[28]=true;
              bERROR|=0x10000000;;
            }

          if( fDMB_Header || fDMB_Trailer )  // F or E  DMB Trailer is missed
            {
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
            if( fCHAMB_ERR[err].find(-1) != fCHAMB_ERR[err].end() )
              {
                fCHAMB_ERR[err].erase(-1);
                fCHAMB_ERR[err].insert(-2);
              }
// Two lines below are commented out because payloads never get filled if 0xA header is missing
//      bCHAMB_PAYLOAD[-2] |= bCHAMB_PAYLOAD[-1];
//      fCHAMB_PAYLOAD[-1] = 0;
          bCHAMB_STATUS[-2] |= bCHAMB_STATUS[-1];
          bCHAMB_STATUS[-1] = 0;
          bCHAMB_ERR[-2] |= bCHAMB_ERR[-1];
          bCHAMB_ERR[-1] = 0;
          bCHAMB_WRN[-2] |= bCHAMB_WRN[-1];
          bCHAMB_WRN[-1] = 0;

          // Chamber id ( DMB_ID + (DMB_CRATE<<4) ) from header
          currentChamber = buf0[1]&0x0FFF;
          ++cntCHAMB_Headers[currentChamber];
          bCHAMB_ERR[currentChamber] |= 0; //Victor's line

          fALCT_Header = false;
          fALCT_Format2007= true;
          fTMB_Header  = false;
          fTMB_Format2007= true;
          uniqueALCT   = true;
          uniqueTMB    = true;

          fTMB_Scope_Start = false;
          fTMB_MiniScope_Start = false;
          fTMB_RPC_Start = false;
          fTMB_BlockedCFEBs_Start = false;

          fTMB_Scope = false;
          fTMB_MiniScope = false;
          fTMB_RPC = false;
          fTMB_BlockedCFEBs = false;


          zeroCounts();
          CFEB_CRC                  = 0;

          nDMBs++;

          dmbBuffers[sourceID][currentChamber] = buf0-4;
          dmbOffsets[sourceID][currentChamber] = buf0-4-buffer_start;
          dmbSize   [sourceID][currentChamber] = 4;

#ifdef LOCAL_UNPACK
          // Print DMB_ID from DMB Header
          COUT<< "Crate=" << setw(3) << setfill('0') << ((buf0[1]>>4)&0x00FF) << " DMB="<<setw(2)<<setfill('0')<<(buf0[1]&0x000F)<<" ";
          // Print ALCT_DAV and TMB_DAV from DMB Header
          //COUT<<setw(1)<<((buf0[0]&0x0020)>>5)<<" "<<((buf0[0]&0x0040)>>6)<<" ";
          COUT<<setw(1)<<((buf0[0]&0x0200)>>9)<<" "<<((buf0[0]&0x0800)>>11)<<" "; //change of format 16.09.05
          // Print CFEB_DAV from DMB Header
          COUT<<setw(1)<<((buf0[0]&0x0010)>>4)<<((buf0[0]&0x0008)>>3)<<((buf0[0]&0x0004)>>2)<<((buf0[0]&0x0002)>>1)<<(buf0[0]&0x0001);
          // Print DMB Header Tag
          COUT << " {";
#endif

          if (fFormat2013) /// 2013 Data format
            {

              // Set variables if we are waiting ALCT, TMB and CFEB records to be present in event
              DAV_ALCT = (buf0[0]&0x0800)>>11;
              DAV_TMB  = (buf0[0]&0x0400)>>10;
              DAV_CFEB = 0;
              if( buf0[0]&0x0001 ) ++DAV_CFEB;
              if( buf0[0]&0x0002 ) ++DAV_CFEB;
              if( buf0[0]&0x0004 ) ++DAV_CFEB;
              if( buf0[0]&0x0008 ) ++DAV_CFEB;
              if( buf0[0]&0x0010 ) ++DAV_CFEB;
              if( buf0[0]&0x0020 ) ++DAV_CFEB;
              if( buf0[0]&0x0040 ) ++DAV_CFEB;
              if( DAV_ALCT ) bCHAMB_PAYLOAD[currentChamber] |= 0x20;
              if( DAV_TMB  ) bCHAMB_PAYLOAD[currentChamber] |= 0x40;

              /// Moved around 7 CFEBs Active and CFEB DAV payload bits to be compatible with 5 CFEBs version
              bCHAMB_PAYLOAD[currentChamber] |= (buf0[0]&0x007f)<<7; 		/// CFEBs DAV
              bCHAMB_PAYLOAD[currentChamber] |= (buf_1[2]&0x001f); 		/// CFEBs Active 5
              bCHAMB_PAYLOAD[currentChamber] |= ((buf_1[2]>>5)&0x0003)<<14;   /// CFEBs Active 6,7
              bCHAMB_STATUS [currentChamber] |= (buf0[0]&0x0080)<<15;		/// CLCT-DAV-Mismatch

            }
          else     /// Pre-2013 DMB Format
            {

              // Set variables if we are waiting ALCT, TMB and CFEB records to be present in event
              DAV_ALCT = (buf0[0]&0x0200)>>9;
              DAV_TMB  = (buf0[0]&0x0800)>>11;
              DAV_CFEB = 0;
              if( buf0[0]&0x0001 ) ++DAV_CFEB;
              if( buf0[0]&0x0002 ) ++DAV_CFEB;
              if( buf0[0]&0x0004 ) ++DAV_CFEB;
              if( buf0[0]&0x0008 ) ++DAV_CFEB;
              if( buf0[0]&0x0010 ) ++DAV_CFEB;
              if( DAV_ALCT ) bCHAMB_PAYLOAD[currentChamber] |= 0x20;
              if( DAV_TMB  ) bCHAMB_PAYLOAD[currentChamber] |= 0x40;
              bCHAMB_PAYLOAD[currentChamber] |= (buf0[0]&0x001f)<<7;
              bCHAMB_PAYLOAD[currentChamber] |=((buf_1[2]>>5)&0x001f);
              bCHAMB_STATUS [currentChamber] |= (buf0[0]&0x0040)<<15;
            }

        }


      // New ALCT data format:
      if( ( buf0[0]==0xDB0A && (buf0[1]&0xF000)==0xD000 && (buf0[2]&0xF000)==0xD000 && (buf0[3]&0xF000)==0xD000)
          &&
          ( (buf_1[0]&0xF000)==0xA000 && (buf_1[1]&0xF000)==0xA000 && (buf_1[2]&0xF000)==0xA000 && (buf_1[3]&0xF000)==0xA000 ) )
        {
          fALCT_Header              = true;
          fALCT_Format2007          = true;
          ALCT_CRC                  = 0;
          ALCT_WordsSinceLastHeader = 4;
          ALCT_WordsSinceLastHeaderZeroSuppressed = 4;

          // Calculate expected number of ALCT words
          ALCT_WordsExpected = 12; // header and trailer always exists

          //  Aauxilary variables
          //   number of wire groups per layer:
          int  nWGs_per_layer = ( (buf1[2]&0x0007) + 1 ) * 16 ;
          // words in the layer
          nWG_round_up   = int(nWGs_per_layer/12)+(nWGs_per_layer%3?1:0);
          //   configuration present:
          bool config_present =  buf1[0]&0x4000;
          //   lct overflow:
          bool lct_overflow   =  buf1[0]&0x2000;
          //   raw overflow:
          bool raw_overflow   =  buf1[0]&0x1000;
          //   l1a_window:
          int  lct_tbins      = (buf1[3]&0x01E0)>>5;
          //   fifo_tbins:
          int  raw_tbins      = (buf1[3]&0x001F);

          ///   Check if ALCT zero suppression enable:
          ALCT_ZSE            = (buf1[1]&0x1000)>>12;

          if (ALCT_ZSE)
            {
              for (int g=0; g<4; g++)
                {
                  if (buf1[g]==0x1000) ALCT_WordsSinceLastHeader -= (nWG_round_up - 1);
                }
            }
#ifdef LOCAL_UNPACK
//        COUT << " Number of Wire Groups: " << nWG_round_up << std::endl;
///       COUT << " ALCT_ZSE: " << ALCT_ZSE << std::endl;
//        COUT << " raw_tbins: " << std::dec << raw_tbins << std::endl;
//        COUT << " LCT Tbins: " << lct_tbins << std::endl;
#endif

          //  Data block sizes:
          //   3 words of Vertex ID register + 5 words of config. register bits:
          int config_size    = ( config_present ? 3 + 5 : 0 );
          //   collision mask register:
          int colreg_size    = ( config_present ? nWGs_per_layer/4 : 0 );
          //   hot channel mask:
          int hot_ch_size    = ( config_present ? nWG_round_up*6 : 0 );
          //   ALCT0,1 (best tracks):
          int alct_0_1_size  = ( !lct_overflow ? 2*lct_tbins : 0 );
          //   raw hit dump size:
          int raw_hit_dump_size=(!raw_overflow ? nWG_round_up*6*raw_tbins : 0 );

#ifdef LOCAL_UNPACK
          // COUT << " Raw Hit Dump: " << std::dec << raw_hit_dump_size << std::endl;
#endif

          ALCT_WordsExpected += config_size + colreg_size + hot_ch_size + alct_0_1_size + raw_hit_dump_size;

#ifdef LOCAL_UNPACK
          COUT<<" <A";
#endif

        }
      else
        {
          // Old ALCT data format

          // == ALCT Header found right after DMB Header
          //   (check for all currently reserved/fixed bits in ALCT first 4 words)
          // if( ( (buf0 [0]&0xF800)==0x6000 && (buf0 [1]&0xFF80)==0x0080 && (buf0 [2]&0xF000)==0x0000 && (buf0 [3]&0xc000)==0x0000 )
          if( ( (buf0 [0]&0xF800)==0x6000 && (buf0 [1]&0x8F80)==0x0080 && (buf0 [2]&0x8000)==0x0000 && (buf0 [3]&0xc000)==0x0000 )
              &&
              ( (buf_1[0]&0xF000)==0xA000 && (buf_1[1]&0xF000)==0xA000 && (buf_1[2]&0xF000)==0xA000 && (buf_1[3]&0xF000)==0xA000 ) )
            {
              fALCT_Header              = true;
              fALCT_Format2007          = false;
              ALCT_CRC                  = 0;
              ALCT_WordsSinceLastHeader = 4;

              // Calculate expected number of ALCT words
              if( (buf0[3]&0x0003)==0 )
                {
                  ALCT_WordsExpected = 12;  // Short Readout
                }

              if( (buf0[1]&0x0003)==1 )  					// Full Readout
                {
                  ALCT_WordsExpected = ((buf0[1]&0x007c) >> 2) *
                                       ( ((buf0[3]&0x0001)   )+((buf0[3]&0x0002)>>1)+
                                         ((buf0[3]&0x0004)>>2)+((buf0[3]&0x0008)>>3)+
                                         ((buf0[3]&0x0010)>>4)+((buf0[3]&0x0020)>>5)+
                                         ((buf0[3]&0x0040)>>6) ) * 12 + 12;
                }
#ifdef LOCAL_UNPACK
              COUT<<" <A";
#endif
            }
        }
#ifdef LOCAL_UNPACK
      //COUT << " ALCT Word Expected: " << ALCT_WordsExpected << std::endl;
#endif

      if( (buf0[0]&0xFFFF)==0xDB0C )
        {

	  // =VB= Handles one of the OTMB corrupted data cases.
	  //      Double TMB data block with 2nd TMB Header is found. 
	  //      Set missing TMB Trailer error.
	  if (fTMB_Header) {
             fERROR[12]=true;        // TMB Trailer is missing
             bERROR   |= 0x1000;
             fCHAMB_ERR[12].insert(currentChamber);
             bCHAMB_ERR[currentChamber] |= 0x1000;
          }

          fTMB_Header              = true;
          fTMB_Format2007          = true;
          TMB_CRC                  = 0;
          TMB_WordsSinceLastHeader = 4;
          TMB_WordsExpected = 0;

          // Calculate expected number of TMB words (whether RPC included will be known later)
          if ( (buf1[1]&0x3000) == 0x3000)
            {
              TMB_WordsExpected = 12;  // Short Header Only
            }
          if ( (buf1[1]&0x3000) == 0x0000)
            {
              TMB_WordsExpected = 48;  // Long Header Only
            }

#ifdef LOCAL_UNPACK
          COUT << " <T";
#endif
        }
      else
        {
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
              fTMB_Format2007          = false;
              TMB_CRC                  = 0;
              TMB_WordsSinceLastHeader = 4;

              // Calculate expected number of TMB words (whether RPC included will be known later)
              if ( (buf0[1]&0x3000) == 0x3000)
                {
                  TMB_WordsExpected = 8;  // Short Header Only
                }
              if ( (buf0[1]&0x3000) == 0x0000)
                {
                  TMB_WordsExpected = 32;  // Long Header Only
                }

              if ( (buf0[1]&0x3000) == 0x1000)
                {
                  // Full Readout   = 28 + (#Tbins * #CFEBs * 6)
                  TMB_Tbins=(buf0[1]&0x001F);
                  TMB_WordsExpected = 28 + TMB_Tbins * ((buf1[0]&0x00E0)>>5) * 6;
                }
#ifdef LOCAL_UNPACK
              COUT << " <T";
#endif
            }
        }
      // New TMB format => very long header Find Firmware revision
      if ( fTMB_Header && fTMB_Format2007 && TMB_WordsSinceLastHeader==8 )
        {
          TMB_Firmware_Revision = buf0[3];
        }

      // New TMB format => very long header
      if ( fTMB_Header && fTMB_Format2007 && TMB_WordsSinceLastHeader==20 )
        {
          // Full Readout   = 44 + (#Tbins * #CFEBs * 6)
          TMB_Tbins=(buf0[3]&0x00F8)>>3;
          TMB_WordsExpected = 44 + TMB_Tbins * (buf0[3]&0x0007) * 6;
        }

      // == ALCT Trailer found
      if(
        // New ALCT data format:
        ( buf0[0]==0xDE0D && (buf0[1]&0xF800)==0xD000 && (buf0[2]&0xF800)==0xD000 && (buf0[3]&0xF000)==0xD000 && fALCT_Format2007 ) ||
        // Old ALCT data format; last check is added to avoid confusion with new TMB header (may not be needed):
        ( (buf0[0]&0x0800)==0x0000 && (buf0[1]&0xF800)==0xD000 && (buf0[2]&0xFFFF)==0xDE0D && (buf0[3]&0xF000)==0xD000 && !fALCT_Format2007 && !(fTMB_Header&&fTMB_Format2007) )
      )
        {
          // should've been (buf0[0]&0xF800)==0xD000 - see comments for sERROR[11]

          // Second ALCT -> Lost both previous DMB Trailer and current DMB Header
          if( !uniqueALCT ) currentChamber = -1;
          // Check if this ALCT record have to exist according to DMB Header
          if(   DAV_ALCT  ) DAV_ALCT = false;
          else DAV_ALCT = true;

          if( !fALCT_Header )
            {
              fERROR[8] = true;
              bERROR   |= 0x100;
              fCHAMB_ERR[8].insert(currentChamber);
              bCHAMB_ERR[currentChamber] |= 0x100;
              fCHAMB_ERR[0].insert(currentChamber);
              bCHAMB_ERR[currentChamber] |= 0x1;
            } // ALCT Header is missing

          if( !fALCT_Format2007 && (buf0[0]&0xF800)!=0xD000 )
            {
              fERROR[11] = true;
              bERROR    |= 0x800;
              fCHAMB_ERR[11].insert(currentChamber);
              bCHAMB_ERR[currentChamber] |= 0x800;
              fCHAMB_ERR[0].insert(currentChamber);
              bCHAMB_ERR[currentChamber] |= 0x1;
            } // some bits in 1st D-Trailer are lost

#ifdef LOCAL_UNPACK
          /// Print Out ALCT word counting
          /*
                COUT << " ALCT Word Since Last Header: " << ALCT_WordsSinceLastHeader << std::endl;
                COUT << " ALCT Word Expected: " << ALCT_WordsExpected << std::endl;
                COUT << " ALCT Word Since Last Header Zero Supressed: " << ALCT_WordsSinceLastHeaderZeroSuppressed << std::endl;
          */
#endif
          /// Check calculated CRC sum against reported
          if( checkCrcALCT )
            {
              uint32_t crc = ( fALCT_Format2007 ? buf0[1] : buf0[0] ) & 0x7ff;
              crc |= ((uint32_t)( ( fALCT_Format2007 ? buf0[2] : buf0[1] ) & 0x7ff)) << 11;
              if( ALCT_CRC != crc )
                {
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
          //ALCT_WordCount = (buf0[3]&0x0FFF);
          CFEB_SampleWordCount = 0;
#ifdef LOCAL_UNPACK
          COUT << "A> ";
#endif
        }

      // Calculation of CRC sum ( algorithm is written by Madorsky )
      if( fALCT_Header && checkCrcALCT )
        {
          for(uint16_t j=0, w=0; j<4; ++j)
            {
              ///w = buf0[j] & 0x7fff;
              w = buf0[j] & (fALCT_Format2007 ? 0xffff : 0x7fff);
              for(uint32_t i=15, t=0, ncrc=0; i<16; i--)
                {
                  t = ((w >> i) & 1) ^ ((ALCT_CRC >> 21) & 1);
                  ncrc = (ALCT_CRC << 1) & 0x3ffffc;
                  ncrc |= (t ^ (ALCT_CRC & 1)) << 1;
                  ncrc |= t;
                  ALCT_CRC = ncrc;
                }
            }
        }

      // == Find Correction for TMB_WordsExpected due to RPC raw hits,
      //    should it turn out to be the new RPC-aware format
      if( fTMB_Header && ((buf0[2]&0xFFFF)==0x6E0B) )
        {
          if (fTMB_Format2007)
            {
	      /* Checks for TMB2007 firmware revisions ranges to detect data format
               * rev.0x50c3 - first revision with changed format 
               * rev.0x42D5 - oldest known from 06/21/2007
               * There is 4-bits year value rollover in revision number (0 in 2016)
               */
              if ((TMB_Firmware_Revision >= 0x50c3) || (TMB_Firmware_Revision < 0x42D5))  
                {
                  // On/off * nRPCs * nTimebins * 2 words/RPC/bin
                  TMB_WordsRPC = ((buf_1[0]&0x0010)>>4) * ((buf_1[0]&0x000c)>>2) * ((buf_1[0]>>5) & 0x1F) * 2;
                }
              else   // original TMB2007 data format (may not work since TMB_Tbins != RPC_Tbins)
                {
                  TMB_WordsRPC = ((buf_1[0]&0x0040)>>6) * ((buf_1[0]&0x0030)>>4) * TMB_Tbins * 2;
                }
            }
          else   // Old format 2006
            {
              TMB_WordsRPC   = ((buf_1[2]&0x0040)>>6) * ((buf_1[2]&0x0030)>>4) * TMB_Tbins * 2;
            }
          TMB_WordsRPC += 2; // add header/trailer for block of RPC raw hits
        }



      // Check for RPC data
      if ( fTMB_Header && (scanbuf(buf0,4, 0x6B04)>=0) )
        {
          fTMB_RPC_Start = true;
        }

      // Check for Scope data
      if ( fTMB_Header && (scanbuf(buf0,4, 0x6B05)>=0) )
        {
          fTMB_Scope_Start = true;
        }

      // Check for Mini-Scope data
      if ( fTMB_Header && (scanbuf(buf0,4, 0x6B07)>=0) )
        {
          fTMB_MiniScope_Start = true;
        }

      // Check for Blocked CFEBs data
      if ( fTMB_Header && (scanbuf(buf0,4, 0x6BCB)>=0) )
        {
          fTMB_BlockedCFEBs_Start = true;
        }


      // Check for end of RPC data
      if ( fTMB_Header && fTMB_RPC_Start
           && (scanbuf(buf0,4, 0x6E04)>=0) )
        {
          fTMB_RPC = true;
        }

      // Check for end of Scope data
      if ( fTMB_Header && fTMB_Scope_Start
           && (scanbuf(buf0,4, 0x6E05)>=0) )
        {
          fTMB_Scope = true;
        }

      // Check for end of Mini-Scope data
      if ( fTMB_Header && fTMB_MiniScope_Start
           && (scanbuf(buf0,4, 0x6E07)>=0) )
        {
          fTMB_MiniScope = true;
        }

      // Check for end of Blocked CFEBs data
      if ( fTMB_Header && fTMB_BlockedCFEBs_Start
           && (scanbuf(buf0,4, 0x6ECB)>=0) )
        {
          fTMB_BlockedCFEBs = true;
        }

      /*
         if ( fTMB_Header && (scanbuf(buf0,4, 0x6E04)>=0) ) {
               TMB_WordsExpected += TMB_WordsRPC;
           }
      */

      // == TMB Trailer found
      if(
        // Old TMB data format; last condition in needed not to confuse if with new ALCT data header
        ((buf0[0]&0xF000)==0xD000 && (buf0[1]&0xF000)==0xD000 && (buf0[2]&0xFFFF)==0xDE0F && (buf0[3]&0xF000)==0xD000 && !fTMB_Format2007 && !(fALCT_Header&&fALCT_Format2007)) ||
        // New TMB data format
        ( buf0[0]==        0xDE0F && (buf0[1]&0xF000)==0xD000 && (buf0[2]&0xF000)==0xD000 && (buf0[3]&0xF000)==0xD000 &&  fTMB_Format2007 )
      )
        {

          // Second TMB -> Lost both previous DMB Trailer and current DMB Header
          if( !uniqueTMB ) currentChamber = -1;
          // Check if this TMB record have to exist according to DMB Header
          if(   DAV_TMB  ) DAV_TMB = false;
          else DAV_TMB = true;

          if(!fTMB_Header)
            {
              fERROR[13] = true;
              bERROR    |= 0x2000;
              fCHAMB_ERR[13].insert(currentChamber);
              bCHAMB_ERR[currentChamber] |= 0x2000;
              fCHAMB_ERR[0].insert(currentChamber);
              bCHAMB_ERR[currentChamber] |= 0x1;
            }  // TMB Header is missing

          // Check calculated CRC sum against reported
          if( checkCrcTMB )
            {
              uint32_t crc = ( fTMB_Format2007 ? buf0[1]&0x7ff : buf0[0]&0x7ff );
              crc |= ((uint32_t)( ( fTMB_Format2007 ? buf0[2]&0x7ff : buf0[1] & 0x7ff ) )) << 11;
              if( TMB_CRC != crc )
                {
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
          //    	2) for extra 4 frames in the new TMB trailer and
          //         for RPC raw hit data, if present
          //
          // If the scope data was enabled in readout, scope data markers (0x6B05
          // and 0x6E05) appear before 0x6E0C, and the optional 0x2AAA and 0x5555
          // trailer words are suppressed.  So far, we only have data with the
          // empty scope content, so more corrections will be needed once
          // non-empty scope data is available. -SV, 5 Nov 2008.
          //
          // If word count is not multiple of 4, add 2 optional words and
          // 4 trailer words.

          int pos = scanbuf(buf_1,4,0x6E0C);
          if (pos==1)
            {
              TMB_WordsExpected += 6;
            }
          // If word count is multiple of 4, add 4 trailer words.
          else if (pos==3)
            {
              TMB_WordsExpected += 4;
            }

          // Correct expected wordcount by RPC data size
          if (fTMB_RPC)
            TMB_WordsExpected += TMB_WordsRPC;

          // Correct expected wordcount by MiniScope data size (22 words + 2 signature words)
          if (fTMB_MiniScope)
            TMB_WordsExpected += 24;

          // Correct expected wordcount by BlockedCFEBs data size (20 words + 2 signature words)
          if (fTMB_BlockedCFEBs)
            TMB_WordsExpected += 22;

          CFEB_SampleWordCount = 0;
#ifdef LOCAL_UNPACK
          COUT << "T> ";
#endif
        }

      if( fTMB_Header && checkCrcTMB )
        {
          for(uint16_t j=0, w=0; j<4; ++j)
            {
              ///w = buf0[j] & 0x7fff;
              w = buf0[j] & (fTMB_Format2007 ? 0xffff : 0x7fff);
              for(uint32_t i=15, t=0, ncrc=0; i<16; i--)
                {
                  t = ((w >> i) & 1) ^ ((TMB_CRC >> 21) & 1);
                  ncrc = (TMB_CRC << 1) & 0x3ffffc;
                  ncrc |= (t ^ (TMB_CRC & 1)) << 1;
                  ncrc |= t;
                  TMB_CRC = ncrc;
                }
            }
        }


      // == CFEB Sample Trailer found

      if( ((buf0[1]&0xF000)==0x7000) &&
          ((buf0[2]&0xF000)==0x7000) &&
          ((buf0[1]!=0x7FFF) || (buf0[2]!=0x7FFF)) &&
          ( ((buf0[3]&0xFFFF)==0x7FFF) ||   // old format
            ( (buf0[3]&buf0[0])==0x0000 && (buf0[3]+buf0[0])==0x7FFF ) // 2007 format
          ) )
        {
#ifdef LOCAL_UNPACK
          if((CFEB_SampleCount%8)  == 0   )
            {
              COUT<<" <";
            }
          if( CFEB_SampleWordCount == 100 )
            {
              COUT<<"+";
            }
#endif
          if( CFEB_SampleWordCount != 100 )
            {
#ifdef LOCAL_UNPACK
              COUT<<"-";
#endif

              fERROR[16] = true;
              bERROR    |= 0x10000;
              fCHAMB_ERR[16].insert(currentChamber);
              bCHAMB_ERR[currentChamber] |= 0x10000;
              fCHAMB_ERR[0].insert(currentChamber);
              bCHAMB_ERR[currentChamber] |= 0x1;
            }

          ++CFEB_SampleCount;

          if( (CFEB_SampleCount%8)==0 )
            {
#ifdef LOCAL_UNPACK
              COUT<<">";
#endif
              CFEB_BSampleCount=0;
              // Count CFEBs
              DAV_CFEB--;
            }

          // Check calculated CRC sum against reported
          if( checkCrcCFEB && CFEB_CRC!=buf0[0] )
            {
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


      // == CFEB B-word found
      if( (buf0[0]&0xF000)==0xB000 && (buf0[1]&0xF000)==0xB000 && (buf0[2]&0xF000)==0xB000 && (buf0[3]&0xF000)==0xB000 )
        {
          bCHAMB_STATUS[currentChamber] |= 0x400000;

#ifdef LOCAL_UNPACK
          if( (CFEB_SampleCount%8)==0 )
            {
              COUT<<" <";
            }
          COUT<<"B";
#endif

          ++CFEB_SampleCount;
          ++CFEB_BSampleCount;

          if( (CFEB_SampleCount%8)==0 )
            {
#ifdef LOCAL_UNPACK
              COUT << ">";
#endif
              CFEB_BSampleCount=0;
              DAV_CFEB--;
            }

          CFEB_SampleWordCount=0;
        }

      // == If it is neither ALCT record nor TMB - probably it is CFEB record and we try to count CRC sum.
      // It very few words of CFEB occasionaly will be misinterpreted as ALCT or TMB header the result
      // for the CRC sum will be wrong, but other errors of Trailers counting will appear as well
      if( checkCrcCFEB && fDMB_Header && !fTMB_Header && !fALCT_Header && CFEB_SampleWordCount )
        for(int pos=0; pos<4; ++pos)
          CFEB_CRC=(buf0[pos]&0x1fff)^((buf0[pos]&0x1fff)<<1)^(((CFEB_CRC&0x7ffc)>>2)|((0x0003&CFEB_CRC)<<13))^((CFEB_CRC&0x7ffc)>>1);


      // == DMB F-Trailer found
      if( (buf0[0]&0xF000)==0xF000 && (buf0[1]&0xF000)==0xF000 && (buf0[2]&0xF000)==0xF000 && (buf0[3]&0xF000)==0xF000 )
        {
          if(!fDMB_Header)
            {
              currentChamber = buf0[3]&0x0FFF;
              fERROR[6] = true;
              bERROR   |= 0x40;
              fCHAMB_ERR[6].insert(currentChamber);
              bCHAMB_ERR[currentChamber] |= 0x40;
              nDMBs++;
              // Set variables if we are waiting ALCT, TMB and CFEB records to be present in event
              if( buf0[0]&0x0400 ) bCHAMB_PAYLOAD[currentChamber] |= 0x20;
              if( buf0[0]&0x0800 ) bCHAMB_PAYLOAD[currentChamber] |= 0x40;
              bCHAMB_PAYLOAD[currentChamber] |= (buf0[0]&0x001f)<<7;
              bCHAMB_PAYLOAD[currentChamber] |=((buf0[0]>>5)&0x1f);

            } // DMB Header is missing
          fDMB_Header  = false;
          fDMB_Trailer = true;
          uniqueALCT   = true;
          uniqueTMB    = true;

          dmbSize[sourceID][currentChamber] = buf0 - dmbBuffers[sourceID][currentChamber];

          // Finally check if DAVs were correct
          checkDAVs();

          // If F-Trailer is lost then do necessary work here
          if( (buf1[0]&0xF000)!=0xE000 || (buf1[1]&0xF000)!=0xE000 || (buf1[2]&0xF000)!=0xE000 || (buf1[3]&0xF000)!=0xE000 )
            {
              for(int err=1; err<nERRORS; ++err)
                if( fCHAMB_ERR[err].find(currentChamber) != fCHAMB_ERR[err].end() )
                  {
                    fCHAMB_ERR[0].insert(currentChamber);
                    bCHAMB_ERR[currentChamber] |= 0x1;
                  }
              // Reset chamber id
              currentChamber=-1;
              /*
                for(int err=0; err<nERRORS; err++)
                if( fCHAMB_ERR[err].find(-1) != fCHAMB_ERR[err].end() )
                fCHAMB_ERR[err].erase(-1);
                bCHAMB_ERR[-1] = 0;
                bCHAMB_WRN[-1] = 0;
              */
            }
#ifdef LOCAL_UNPACK
          // Print DMB F-Trailer marker
          COUT << " }";
#endif
        }

      // == DMB E-Trailer found
      if( (buf0[0]&0xF000)==0xE000 && (buf0[1]&0xF000)==0xE000 && (buf0[2]&0xF000)==0xE000 && (buf0[3]&0xF000)==0xE000 )
        {
          if( !fDMB_Header && !fDMB_Trailer ) nDMBs++; // both DMB Header and DMB F-Trailer were missing

          if (fFormat2013) /// 2013 Data format
            {
              ///!!! Put correct bits positions
              bCHAMB_STATUS[currentChamber] |= (buf0[0]&0x0800)>>11;        /// ALCT FIFO FULL
              bCHAMB_STATUS[currentChamber] |= (buf0[0]&0x0400)>>9;         /// TMB FIFO Full
              bCHAMB_STATUS[currentChamber] |= (buf0[0]&0x0080)<<8;         /// TMB End Timeout

              if( fDMB_Trailer )   // F-Trailer exists
                {
                  bCHAMB_STATUS[currentChamber] |= (buf_1[2]&0x0E00)>>7;      /// CFEB 1-3 FIFO Full
                  bCHAMB_STATUS[currentChamber] |= (buf_1[3]&0x0003)<<3;      /// CFEB 4-5 FIFO Full
                  bCHAMB_STATUS[currentChamber] |= (buf_1[3]&0x000C)<<21;     /// CFEB 6-7 FIFO Full
                  bCHAMB_STATUS[currentChamber] |= (buf_1[3]&0x0800)>>4;      /// ALCT Start Timeout
                  bCHAMB_STATUS[currentChamber] |= (buf_1[2]&0x0100);         /// TMB Start Timeout
                  bCHAMB_STATUS[currentChamber] |= (buf_1[3]&0x01f0)<<5;      /// CFEB 1-5 Start Timeout
                  bCHAMB_STATUS[currentChamber] |= (buf_1[3]&0x0600)<<16;     /// CFEB 6-7 Start Timeout
                  bCHAMB_STATUS[currentChamber] |= (buf_1[0]&0x0800)<<3;      /// ALCT End Timeout
                  bCHAMB_STATUS[currentChamber] |= (buf_1[1]&0x001f)<<16;     /// CFEB 1-5 End Timeout
                  bCHAMB_STATUS[currentChamber] |= (buf_1[1]&0x0060)<<21;     /// CFEB 6-7 End Timeout

                }

            }
          else
            {
              bCHAMB_STATUS[currentChamber] |= (buf0[0]&0x0800)>>11;	/// ALCT FIFO FULL
              bCHAMB_STATUS[currentChamber] |= (buf0[0]&0x0400)>>9;		/// TMB FIFO Full
              bCHAMB_STATUS[currentChamber] |= (buf0[0]&0x03E0)>>3;		/// CFEB 1-5 FIFO Full

              if( fDMB_Trailer )   // F-Trailer exists
                {
                  bCHAMB_STATUS[currentChamber] |= (buf_1[2]&0x0002)<<6;	/// ALCT Start Timeout
                  bCHAMB_STATUS[currentChamber] |= (buf_1[2]&0x0001)<<8;	/// TMB Start Timeout
                  bCHAMB_STATUS[currentChamber] |= (buf_1[3]&0x001f)<<9;	/// CFEB 1-5 Start Timeout
                  bCHAMB_STATUS[currentChamber] |= (buf_1[3]&0x0040)<<8;	/// ALCT End Timeout
                  bCHAMB_STATUS[currentChamber] |= (buf_1[3]&0x0020)<<10;	/// TMB End Timeout
                  bCHAMB_STATUS[currentChamber] |= (buf_1[3]&0x0f80)<<9;	/// CFEB 1-5 End Timeout
                }

            }
          fDMB_Header  = false;

          // If chamber id is unknown it is time to find it out
          if( currentChamber==-1 )
            {
              currentChamber = buf0[1]&0x0FFF;
              for(int err=0; err<nERRORS; ++err)
                if( fCHAMB_ERR[err].find(-1) != fCHAMB_ERR[err].end() )
                  {
                    fCHAMB_ERR[err].insert(currentChamber);
                    fCHAMB_ERR[err].erase(-1);
                  }
              bCHAMB_STATUS[currentChamber] = bCHAMB_STATUS[-1];
              bCHAMB_STATUS[-1] = 0;
              bCHAMB_ERR[currentChamber] = bCHAMB_ERR[-1];
              bCHAMB_ERR[-1] = 0;
              bCHAMB_WRN[currentChamber] = bCHAMB_WRN[-1];
              bCHAMB_WRN[-1] = 0;
            }
          ++cntCHAMB_Trailers[buf0[1]&0x0FFF];

          dmbSize[sourceID][currentChamber] = buf0 - dmbBuffers[sourceID][currentChamber];

          // Lost DMB F-Trailer before
          if( !fDMB_Trailer )
            {
              fERROR[6] = true;
              bERROR   |= 0x40;
              fCHAMB_ERR[6].insert(currentChamber);
              bCHAMB_ERR[currentChamber] |= 0x40;
              fCHAMB_ERR[0].insert(currentChamber);
              bCHAMB_ERR[currentChamber] |= 0x1;
              // Check if DAVs were correct here
              checkDAVs();
            }
          fDMB_Trailer = false;

#ifdef LOCAL_UNPACK
          // Print DMB E-Trailer marker
          COUT<<" DMB="<<(buf0[1]&0x000F);
          COUT << "; "
               << ALCT_WordsSinceLastHeader << "-"
               << ALCT_WordCount << "-"
               << ALCT_WordsExpected
               << "      "
               << TMB_WordsSinceLastHeader << "-"
               << TMB_WordCount << "-"
               << TMB_WordsExpected
               << endl;
#endif

          checkTriggerHeadersAndTrailers();

          //
          for(int err=0; err<nERRORS; ++err)
            if( fCHAMB_ERR[err].find(-1) != fCHAMB_ERR[err].end() )
              {
                fCHAMB_ERR[err].erase(-1);
                fCHAMB_ERR[err].insert(-2);
              }
          bCHAMB_STATUS[-2] |= bCHAMB_STATUS[-1];
          bCHAMB_STATUS[-1] = 0;
          bCHAMB_ERR[-2] |= bCHAMB_ERR[-1];
          bCHAMB_ERR[-1] = 0;
          bCHAMB_WRN[-2] |= bCHAMB_WRN[-1];
          bCHAMB_WRN[-1] = 0;

          if( currentChamber != -1 )
            for(int err=1; err<nERRORS; ++err)
              if( fCHAMB_ERR[err].find(currentChamber) != fCHAMB_ERR[err].end() )
                {
                  fCHAMB_ERR[0].insert(currentChamber);
                  bCHAMB_ERR[currentChamber] |= 0x1;
                }

          currentChamber=-1;
#ifdef LOCAL_UNPACK
          /*
                // Print DMB E-Trailer marker
                COUT<<" DMB="<<(buf0[1]&0x000F);
                COUT << "; "
          	   << ALCT_WordsSinceLastHeader << "-"
          	   << ALCT_WordCount << "-"
          	   << ALCT_WordsExpected
          	   << "      "
          	   << TMB_WordsSinceLastHeader << "-"
          	   << TMB_WordCount << "-"
          	   << TMB_WordsExpected
          	   << endl;
          */
#endif
        }

      // == DDU Trailer found
      if( buf0[0]==0x8000 && buf0[1]==0x8000 && buf0[2]==0xFFFF && buf0[3]==0x8000 )
        {

/////////////////////////////////////////////////////////
          checkDAVs();

          checkTriggerHeadersAndTrailers();

/////////////////////////////////////////////////////////

          if( DDU_WordsSinceLastHeader>3 && !nDMBs )
            {
              fERROR[28]=true;
              bERROR|=0x10000000;;
            }

          if(fDDU_Trailer)
            {
              fERROR[2] = true;
              bERROR   |= 0x4;
            } // DDU Header is missing
          fDDU_Trailer=true;
          fDDU_Header=false;

          if( fDMB_Header || fDMB_Trailer )
            {
#ifdef LOCAL_UNPACK
              COUT << " Ex-Err: DMB (Header, Trailer) " << std::endl;
#endif
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
            if( fCHAMB_ERR[err].find(-1) != fCHAMB_ERR[err].end() )
              {
                fCHAMB_ERR[err].erase(-1);
                fCHAMB_ERR[err].insert(-2);
              }
          bCHAMB_STATUS[-2] |= bCHAMB_STATUS[-1];
          bCHAMB_STATUS[-1] = 0;
          bCHAMB_ERR[-2] |= bCHAMB_ERR[-1];
          bCHAMB_ERR[-1] = 0;
          bCHAMB_WRN[-2] |= bCHAMB_WRN[-1];
          bCHAMB_WRN[-1] = 0;

          for(int err=1; err<nERRORS; ++err)
            if( fCHAMB_ERR[err].find(-2) != fCHAMB_ERR[err].end() )
              {
                fCHAMB_ERR[0].insert(-2);
                bCHAMB_ERR[-2] |= 0x1;
              }

          dduSize[sourceID] = buf0-dduBuffers[sourceID];

          ++cntDDU_Trailers; // Increment DDUTrailer counter

          // == Combining 2 words into 24bit value
          DDU_WordCount = buf2[2] | ((buf2[3] & 0xFF) <<16) ;

          if( (DDU_WordsSinceLastHeader+4) != DDU_WordCount )
            {
              fERROR[4] = true;
              bERROR   |= 0x10;
            }

          if( DMB_Active!=nDMBs )
            {
              fERROR[24] = true;
              bERROR    |= 0x1000000;
            }

#ifdef LOCAL_UNPACK
          COUT<<"DDU Trailer Occurrence "<<cntDDU_Trailers<<endl;
          COUT<<"----------------------------------------------------------"<<endl;
          COUT<<"DDU 64-bit words = Actual - DDUcounted ="<<DDU_WordsSinceLastHeader+4<<"-"<<DDU_WordCount<<endl;
#endif

          // increment statistics Errors and Warnings (i=0 case is handled in DDU Header)
          for(int err=1; err<nERRORS; ++err)
            {
              if( fERROR[err] )
                {
                  fERROR[0] = true;
                  bERROR |= 0x1;
#ifdef LOCAL_UNPACK
                  CERR<<"\nDDU Header Occurrence = "<<cntDDU_Headers;
                  CERR<<"  ERROR "<<err<<"  " <<sERROR[err]<<endl;
#endif
                }
            }

#ifdef LOCAL_UNPACK
          for(int wrn=1; wrn<nWARNINGS; ++wrn)
            {
              if( fWARNING[wrn] )
                {
                  COUT<<"\nDDU Header Occurrence = "<<cntDDU_Headers;
                  COUT<<"  WARNING "<<wrn<<"  "<<sWARNING[wrn]<<endl;
                }
            }
#endif

          bDDU_ERR[sourceID] |= bERROR;
          bDDU_WRN[sourceID] |= bWARNING;
          sync_stats();

          DDU_WordsSinceLastHeader=0;
          DDU_WordsSinceLastTrailer=0;
          if (modeDDUonly)
            {
              buffer+=4;
              buf_1 = &(tmpbuf[0]);  // Just for safety
              buf0  = &(tmpbuf[4]);  // Just for safety
              buf1  = &(tmpbuf[8]);  // Just for safety
              buf2  = &(tmpbuf[12]); // Just for safety
              bzero(tmpbuf, sizeof(uint16_t)*16);
              return length-4;
            }
        }

      if (!modeDDUonly)
        {
          // DCC Trailer 1 && DCC Trailer 2
          // =VB= Added support for Sep. 2008 CMS DAQ DCC format
          // =VB= 04.18.09 Removed (buf2[0]&0x0003) == 0x3 check for old DCC format to satisfy older format of simulated data
          if( (buf1[3]&0xFF00) == 0xEF00 &&
              ( ((buf2[3]&0xFF00) == 0xAF00 )
                ||
                (( buf2[3]&0xFF00) == 0xA000 && (buf2[0]&0x0003) == 0x0) ) )
            {
              // =VB= Added check that there is no DCCHeader detected to set missing DCC Header error
              if(!fDCC_Header || fDCC_Trailer)
                {
                  fERROR[26] = true;
                  bERROR|=0x4000000;
                  fERROR[0] = true;
                  bERROR|=0x1;
                } // DCC Header is missing
              fDCC_Trailer=true;
              fDCC_Header=false;

              if( fDDU_Header )
                {
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
              bzero(tmpbuf, sizeof(uint16_t)*16);
              sync_stats();
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

  if (!modeDDUonly && !fDCC_Trailer && !fDCC_Header)
    {
      fERROR[26] = true;
      bERROR|=0x4000000;
      fERROR[25] = true;
      bERROR|=0x2000000;
      fERROR[0]=true;
      bERROR|=0x1;
      sync_stats();
      return length;

    }

  return -2;
}


void CSCDCCExaminer::clear()
{
  bzero(fERROR,   sizeof(bool)*nERRORS);
  bzero(fWARNING, sizeof(bool)*nWARNINGS);
  bzero(fSUM_ERROR,   sizeof(bool)*nERRORS);
  bzero(fSUM_WARNING, sizeof(bool)*nWARNINGS);
  bERROR = 0;
  bWARNING = 0;
  bSUM_ERROR = 0;
  bSUM_WARNING = 0;
  for(int err=0; err<nERRORS;   ++err) fCHAMB_ERR[err].clear();
  for(int wrn=0; wrn<nWARNINGS; ++wrn) fCHAMB_WRN[wrn].clear();
  bCHAMB_ERR.clear();
  bCHAMB_WRN.clear();
  bCHAMB_PAYLOAD.clear();
  bCHAMB_STATUS.clear();
  bDDU_ERR.clear();
  bDDU_WRN.clear();
  dduBuffers.clear();
  dduOffsets.clear();
  dmbBuffers.clear();
  dmbOffsets.clear();
  dduSize.clear();
  dmbSize.clear();
}


void CSCDCCExaminer::zeroCounts()
{
  ALCT_WordsSinceLastHeader = 0;
  ALCT_WordsSinceLastHeaderZeroSuppressed =0;
  ALCT_WordCount            = 0;
  ALCT_WordsExpected        = 0;
  ALCT_ZSE                  = 0;
  TMB_WordsSinceLastHeader  = 0;
  TMB_WordCount             = 0;
  TMB_WordsExpected         = 0;
  TMB_Tbins                 = 0;
  CFEB_SampleWordCount      = 0;
  CFEB_SampleCount          = 0;
  CFEB_BSampleCount         = 0;
}


void CSCDCCExaminer::checkDAVs()
{
  if( DAV_ALCT )
    {
      fERROR[21] = true;
      bERROR   |= 0x200000;
      fCHAMB_ERR[21].insert(currentChamber);
      bCHAMB_ERR[currentChamber] |= 0x200000;
      DAV_ALCT = false;
    }
  if( DAV_TMB  )
    {
      fERROR[22] = true;
      bERROR   |= 0x400000;
      fCHAMB_ERR[22].insert(currentChamber);
      bCHAMB_ERR[currentChamber] |= 0x400000;
      DAV_TMB = false;
    }
  if( DAV_CFEB && DAV_CFEB!=-16)
    {
      fERROR[23] = true;
      bERROR   |= 0x800000;
      fCHAMB_ERR[23].insert(currentChamber);
      bCHAMB_ERR[currentChamber] |= 0x800000;
      DAV_CFEB = 0;
    }
}


void CSCDCCExaminer::checkTriggerHeadersAndTrailers()
{
#ifdef LOCAL_UNPACK
  /*
  COUT << " Ex-ALCT-Word-count " << std::endl;
  COUT << " ALCT Words Since Last Header: " <<  ALCT_WordsSinceLastHeader << std::endl;
  COUT << " ALCT Word Count: " <<  ALCT_WordCount << std::endl;
  COUT << " ALCT Words Expected: " << ALCT_WordsExpected << std::endl;
  */
#endif
  if( !fALCT_Header && ( ALCT_WordsSinceLastHeader!=ALCT_WordCount || ALCT_WordsSinceLastHeader!=ALCT_WordsExpected )
      && ALCT_ZSE==0 )
    {
      fERROR[9] = true;
      bERROR   |= 0x200;
      fCHAMB_ERR[9].insert(currentChamber);
      bCHAMB_ERR[currentChamber] |= 0x200;
      ALCT_WordsSinceLastHeader = 0;
      ALCT_WordCount            = 0;
      ALCT_WordsSinceLastHeader = 0;
      ALCT_WordsExpected        = 0;
    } // ALCT Word Count Error

  if( !fALCT_Header && (ALCT_WordsSinceLastHeader!=ALCT_WordsExpected
                        || ALCT_WordsSinceLastHeaderZeroSuppressed!=ALCT_WordCount) && ALCT_ZSE!=0 )
    {
      fERROR[9] = true;
      bERROR   |= 0x200;
      fCHAMB_ERR[9].insert(currentChamber);
      bCHAMB_ERR[currentChamber] |= 0x200;
      ALCT_WordsSinceLastHeaderZeroSuppressed =0;
      ALCT_WordsSinceLastHeader = 0;
      ALCT_WordCount            = 0;
      ALCT_WordsSinceLastHeader = 0;
      ALCT_WordsExpected        = 0;
    } // ALCT Word Count Error With zero suppression

  if( !fTMB_Header && ( TMB_WordsSinceLastHeader!=TMB_WordCount || TMB_WordsSinceLastHeader!=TMB_WordsExpected ) )
    {
      fERROR[14] = true;
      bERROR    |= 0x4000;
      fCHAMB_ERR[14].insert(currentChamber);
      bCHAMB_ERR[currentChamber] |= 0x4000;
      TMB_WordsSinceLastHeader = 0;
      TMB_WordCount            = 0;
      TMB_WordsSinceLastHeader = 0;
      TMB_WordsExpected        = 0;
    } // TMB Word Count Error

  if( (CFEB_SampleCount%8)!=0 )
    {
      fERROR[17] = true;
      bERROR    |= 0x20000;
      fCHAMB_ERR[17].insert(currentChamber);
      bCHAMB_ERR[currentChamber] |= 0x20000;
      CFEB_SampleCount = 0;
    } // Number of CFEB samples != 8*n

  if(fALCT_Header)
    {
      fERROR[7] = true;  // ALCT Trailer is missing
      bERROR   |= 0x80;
      fCHAMB_ERR[7].insert(currentChamber);
      bCHAMB_ERR[currentChamber] |= 0x80;
      ALCT_WordsSinceLastHeaderZeroSuppressed =0;
      ALCT_WordsSinceLastHeader = 0;
      ALCT_WordsExpected        = 0;
      fALCT_Header = 0;
    }

  if(fTMB_Header)
    {
      fERROR[12]=true;        // TMB Trailer is missing
      bERROR   |= 0x1000;
      fCHAMB_ERR[12].insert(currentChamber);
      bCHAMB_ERR[currentChamber] |= 0x1000;
      TMB_WordsSinceLastHeader = 0;
      TMB_WordsExpected        = 0;
      fTMB_Header = false;
    }
}

inline void CSCDCCExaminer::sync_stats()
{
  for (int err=0; err<nERRORS; ++err)
    fSUM_ERROR[err] |= fERROR[err];
  for (int wrn=0; wrn<nWARNINGS; ++wrn)
    fSUM_WARNING[wrn] |= fWARNING[wrn];
  bSUM_ERROR            |= bERROR;
  bSUM_WARNING  |= bWARNING;
}

inline int CSCDCCExaminer::scanbuf(const uint16_t* &buffer, int32_t length, uint16_t sig, uint16_t mask)
{
  for (int i=0; i<length; i++)
    {
      if ( (buffer[i]&mask) == sig)
        {
          return i;
        }
    }
  return -1;
}


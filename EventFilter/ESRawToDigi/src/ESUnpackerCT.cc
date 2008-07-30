#include "EventFilter/ESRawToDigi/interface/ESUnpackerCT.h"
#include "EventFilter/ESRawToDigi/interface/ESCrcKchipFast.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/EcalDetId/interface/EcalDetIdCollections.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"

ESUnpackerCT::ESUnpackerCT(const ParameterSet& ps) 
  : pset_(ps), fedId_(0), run_number_(0), orbit_number_(0), bx_(0), lv1_(0), trgType_(0)
{

  debug_ = pset_.getUntrackedParameter<bool>("debugMode", false);

}

ESUnpackerCT::~ESUnpackerCT() {
}

void ESUnpackerCT::interpretRawData(int fedId, const FEDRawData & rawData, ESRawDataCollection & dccs, ESLocalRawDataCollection & kchips, ESDigiCollection & digis) {
  
  int nWords = rawData.size()/sizeof(Word64);
  if (nWords==0) return;
  int dccWords = 6;
  
  // Event header
  const Word64* header = reinterpret_cast<const Word64* >(rawData.data()); --header;
  bool moreHeaders = true;
  while (moreHeaders) {
    ++header;
    FEDHeader ESHeader( reinterpret_cast<const unsigned char*>(header) );
    if ( !ESHeader.check() ) break; // throw exception?
    if ( ESHeader.sourceID() != fedId) throw cms::Exception("PROBLEM in ESUnpackerCT !");

    fedId_ = ESHeader.sourceID();
    lv1_   = ESHeader.lvl1ID();
    bx_    = ESHeader.bxID();

    if (debug_) {
      cout<<"[ESUnpackerCT]: FED Header candidate. Is header? "<< ESHeader.check();
      if (ESHeader.check())
        cout <<". BXID: "<<bx_<<" SourceID : "<<fedId_<<" L1ID: "<<lv1_<<endl;
      else cout<<" WARNING!, this is not a ES Header"<<endl;
    }

    moreHeaders = ESHeader.moreHeaders();
  }

  // Event trailer
  const Word64* trailer = reinterpret_cast<const Word64* >(rawData.data())+(nWords-1); ++trailer;
  bool moreTrailers = true;
  while (moreTrailers) {
    --trailer;
    FEDTrailer ESTrailer(reinterpret_cast<const unsigned char*>(trailer));
    if ( !ESTrailer.check()) { ++trailer; break; } // throw exception?
    if ( ESTrailer.lenght()!= nWords) throw cms::Exception("PROBLEM in ESUnpackerCT !!");

    if (debug_)  {
      cout<<"[ESUnpackerCT]: FED Trailer candidate. Is trailer? "<<ESTrailer.check();
      if (ESTrailer.check())
        cout<<". Length of the ES event: "<<ESTrailer.lenght()<<endl;
      else cout<<" WARNING!, this is not a ES Trailer"<<endl;
    }

    moreTrailers = ESTrailer.moreTrailers();
  }

  if (fedId == 1) {

    DCCHeader.clear();
    for (const Word64* word=(header+1); word!=trailer; ++word) {
      if (debug_) cout<<"TLS   : "<<print(*word)<<endl;
      DCCHeader.push_back(*word);
    }
    word2TLS(DCCHeader);  // word to top level supervisor

    ESDCCHeaderBlock ESTLS;
    ESTLS.setFedId(fedId);
    ESTLS.setPacketLength(packetLen_);
    ESTLS.setMajorVersion(vmajor_);
    ESTLS.setBMMeasurements(BMMeasurements_);
    ESTLS.setBeginOfSpillSec(beginOfSpillSec_);
    ESTLS.setBeginOfSpillMilliSec(beginOfSpilliMilliSec_);
    ESTLS.setEndOfSpillSec(endOfSpillSec_);
    ESTLS.setEndOfSpillMilliSec(endOfSpilliMilliSec_);
    ESTLS.setBeginOfSpillLV1(beginOfSpillLV1_);
    ESTLS.setEndOfSpillLV1(endOfSpillLV1_);
    
    dccs.push_back(ESTLS);

    if (debug_) cout<<packetLen_<<" "<<vmajor_<<" "<<BMMeasurements_<<" "<<runNum_<<" "<<beginOfSpillSec_<<" "<<beginOfSpilliMilliSec_<<" "<<endOfSpillSec_<<" "<<endOfSpilliMilliSec_<<" "<<beginOfSpillLV1_<<" "<<endOfSpillLV1_<<endl;
  }

  if (fedId == 4) {

    DCCHeader.clear();
    for (const Word64* word=(header+1); word!=trailer; ++word) {
      if (debug_) cout<<"CTS   : "<<print(*word)<<endl;
      DCCHeader.push_back(*word);
    }
    word2CTS(DCCHeader);

    ESDCCHeaderBlock ESCTS;
    ESCTS.setFedId(fedId);
    ESCTS.setPacketLength(packetLen_);
    ESCTS.setMajorVersion(vmajor_);
    ESCTS.setMinorVersion(vminor_);
    ESCTS.setTimeStampSec(timestamp_sec_);
    ESCTS.setTimeStampUSec(timestamp_usec_);
    ESCTS.setSpillNumber(spillNum_);
    ESCTS.setRunNumber(runNum_);
    ESCTS.setEventInSpill(evtInSpill_);
    ESCTS.setEV(ev_);
    ESCTS.setCAMACError(camacErr_);
    ESCTS.setVMEError(vmeErr_);
    vector<int> ADCch_status;
    vector<int> ADCch;
    for (unsigned int i=0; i<12; ++i) {
      ADCch_status.push_back(ADCchStatus_[i]);
      ADCch.push_back(ADCch_[i]);
    }
    ESCTS.setADCChannelStatus(ADCch_status);
    ESCTS.setADCChannel(ADCch);
    vector<int> TDC_status;
    vector<int> TDC;
    for (unsigned int i=0; i<8; ++i) {
      TDC_status.push_back(TDCStatus_[i]);
      TDC.push_back(TDC_[i]);
    }
    ESCTS.setTDCChannelStatus(TDC_status);
    ESCTS.setTDCChannel(TDC);

    dccs.push_back(ESCTS);
  }

  if (fedId>=10 && fedId<=13) {

    DCCHeader.clear();
    for (const Word64* word=(header+1); word!=trailer; ++word) {
      if (debug_) cout<<"Crepe   : "<<print(*word)<<endl;
      DCCHeader.push_back(*word);
    }
    word2Crepe(DCCHeader);

    ESDCCHeaderBlock ESCrepe;
    ESCrepe.setFedId(fedId);
    ESCrepe.setLV1(lv1_);
    ESCrepe.setBX(bx_);
    ESCrepe.setPacketLength(packetLen_);
    ESCrepe.setRunNumber(runNum_);
    ESCrepe.setMajorVersion(vmajor_);
    ESCrepe.setMinorVersion(vminor_);
    ESCrepe.setBC(bc_);
    ESCrepe.setEV(ev_);

    dccs.push_back(ESCrepe); 
  }

  if (fedId>=40 && fedId<=43) {
    
    // DCC data
    DCCHeader.clear();
    for (const Word64* word=(header+1); word!=(header+dccWords+1); ++word) {
      if (debug_) cout<<"DCC   : "<<print(*word)<<endl;
      DCCHeader.push_back(*word);
    }
    word2DCCHeader(DCCHeader);

    ESDCCHeaderBlock ESDCCHeader;
    ESDCCHeader.setFedId(fedId_);
    ESDCCHeader.setLV1(lv1_);
    ESDCCHeader.setBX(bx_);    
    ESDCCHeader.setEventLength(evtLen_);
    ESDCCHeader.setDCCErrors(DCCErr_);
    ESDCCHeader.setRunNumber(runNum_);
    ESDCCHeader.setRunType(runType_);
    ESDCCHeader.setTriggerType(trgType_);
    ESDCCHeader.setCompressionFlag(compFlag_);
    ESDCCHeader.setOrbitNumber(orbit_);
    ESDCCHeader.setMajorVersion(vmajor_);
    ESDCCHeader.setMinorVersion(vminor_);
    ESDCCHeader.setOptoRX0(optoRX0_);
    ESDCCHeader.setOptoRX1(optoRX1_);
    ESDCCHeader.setOptoRX2(optoRX2_);
    vector<int> enabledFiber;
    vector<int> FEch_status;
    for (unsigned int i=0; i<36; ++i) {
      enabledFiber.push_back(i);
      FEch_status.push_back(FEch_[i]);
    }
    ESDCCHeader.setFEChannelStatus(FEch_status);
    
    dccs.push_back(ESDCCHeader);  
    
    // Event data
    map<int, vector<Word16> > map_data;
    map_data.clear();
    static const Word64 WORD16_mask = 0xffff;
    Word16 kword;
    int count = 0;
    int kchip = 0;
    for (const Word64* word=(header+dccWords+1); word!=trailer; ++word) {
      if (debug_) cout<<"Event : "<<print(*word)<<endl;

      kchip = count/298;
        
      kword = *word       & WORD16_mask;
      map_data[kchip*4+0].push_back(kword);
      
      kword = *word >> 16 & WORD16_mask;
      map_data[kchip*4+1].push_back(kword);
      
      kword = *word >> 32 & WORD16_mask;
      map_data[kchip*4+2].push_back(kword);
      
      kword = *word >> 48 & WORD16_mask;
      map_data[kchip*4+3].push_back(kword);

      count++;
    }

    count = 0;
    map<int, vector<Word16> >::const_iterator kit;
    for (kit=map_data.begin(); kit!=map_data.end(); ++kit) {
      word2digi(enabledFiber[kit->first], kit->second, kchips, digis);
    }
  }

}

void ESUnpackerCT::word2DCCHeader(const vector<Word64> & word) {

  evtLen_   = (word[0])       & 0xffffff;
  DCCErr_   = (word[0] >> 24) & 0x00ff;
  runNum_   = (word[0] >> 32) & 0xffffff;

  runType_  = (word[1])       & 0xffffffff;
  trgType_  = (word[1] >> 32) & 0xffff;
  compFlag_ = (word[1] >> 48) & 0x00ff;

  orbit_    = (word[2])       & 0xffffffff;
  vminor_   = (word[2] >> 40) & 0x00ff;
  vmajor_   = (word[2] >> 48) & 0x00ff;

  optoRX0_  = (word[3] >> 48) & 0x00ff;
  optoRX1_  = (word[4] >> 48) & 0x00ff;
  optoRX2_  = (word[5] >> 48) & 0x00ff;

  for (unsigned int i=0; i<3; ++i) 
    for (unsigned int j=0; j<12; ++j) 
      FEch_[i*12+j] = (word[i+3] >> (j*4)) & 0x000f;      
  
}

void ESUnpackerCT::word2CTS(const vector<Word64> & word) {

  packetLen_       = (word[0])       & 0xffff;

  vminor_          = (word[1] >> 16) & 0x00ff;
  vmajor_          = (word[1] >> 24) & 0x00ff;
  timestamp_sec_   = (word[1] >> 32) & 0xffffffff;

  timestamp_usec_  = (word[2])       & 0xffffffff;
  lv1_             = (word[2] >> 32) & 0xffffffff;

  spillNum_        = (word[3])       & 0xffff;
  runNum_          = (word[3] >> 16) & 0xffff;
  evtInSpill_      = (word[3] >> 32) & 0xffff;

  ev_              = (word[4])       & 0xffffffff;
  camacErr_        = (word[4] >> 32) & 0xffff;
  vmeErr_          = (word[4] >> 48) & 0xffff;

  exRunNum_        = (word[5])       & 0xffffffff;

  if (debug_) {
    cout<<packetLen_<<endl;
    cout<<vminor_<<" "<<vmajor_<<" "<<timestamp_sec_<<endl;
    cout<<timestamp_usec_<<" "<<lv1_<<endl;
    cout<<spillNum_<<" "<<runNum_<<" "<<evtInSpill_<<endl;
    cout<<ev_<<" "<<camacErr_<<" "<<vmeErr_<<endl;
    cout<<exRunNum_<<endl;
  }

  for (unsigned int i=0; i<6; ++i) {
    for (unsigned int j=0; j<2; ++j) { 
      ADCch_[i*2+j]       = (word[i+6] >> (j*32))    & 0xffffff;
      ADCchStatus_[i*2+j] = (word[i+6] >> (24+j*32)) & 0xff;  
      if (debug_) cout<<ADCch_[i*2+j]<<" "<<ADCchStatus_[i*2+j]<<endl;
    }
  }

  for (unsigned int i=0; i<4; ++i) { 
    for (unsigned int j=0; j<2; ++j) { 
      TDC_[i*2+j]       = (word[i+12] >> (j*32))    & 0xffffff;
      TDCStatus_[i*2+j] = (word[i+12] >> (24+j*32)) & 0xff;  
      if (debug_) cout<<TDC_[i*2+j]<<" "<<TDCStatus_[i*2+j]<<endl;
    }
  }

}

void ESUnpackerCT::word2Crepe(const vector<Word64> & word) {

  packetLen_ = (word[0])       & 0xffff;
  
  runNum_    = (word[1])       & 0xffffffff;
  lv1_       = (word[1] >> 32) & 0xfffffff;

  vminor_    = (word[2])       & 0xffff;
  vmajor_    = (word[2] >> 16) & 0xffff;  
  bc_        = (word[2] >> 32) & 0xffff;
  ev_        = (word[2] >> 48) & 0xffff;

}

void ESUnpackerCT::word2TLS(const vector<Word64> & word) {

  packetLen_             = (word[0])       & 0xffff;

  vmajor_                = (word[1])       & 0x00ff;
  BMMeasurements_        = (word[1] >>  8) & 0x00ff;
  runNum_                = (word[1] >> 32) & 0xffff;

  beginOfSpillSec_       = (word[2])       & 0xffff;
  beginOfSpilliMilliSec_ = (word[2] >> 32) & 0xffff;
    
  endOfSpillSec_         = (word[3])       & 0xffff;
  endOfSpilliMilliSec_   = (word[3] >> 32) & 0xffff;

  beginOfSpillLV1_       = (word[4])       & 0xffff;
  endOfSpillLV1_         = (word[4] >> 32) & 0xffff;

}

void ESUnpackerCT::word2digi(int fiber, const vector<Word16> & word, ESLocalRawDataCollection & kchips, ESDigiCollection & digis) 
{                

  //for (int i=0; i<word.size(); ++i) cout<<"Fiber : "<<fiber<<" "<<print(word[i])<<endl;

  if (word.size() != 298) {
    cout<<"KChip data length is not 298 for fiber : "<<fiber<<endl;
    return;
  }
  
  int kBC = word[0] & 0x0fff; 
  int kEC = word[1] & 0x00ff;
  int kID = (word[2] >> 8) & 0x00ff; 
  int kFlag1 = (word[0] >> 12) & 0x000f;
  int kFlag2 = (word[1] >>  8) & 0x00ff; 
  int chksum = word[297] & 0xffff;

  ESCrcKchipFast crcChecker;

  uint32_t packet_length = (kFlag1 & 0x07) ? 5 : 299 ; 

  for(uint32_t kk=0; kk < (packet_length-1); ++kk) crcChecker.add((unsigned int) word[kk]); 

  ESKCHIPBlock ESKCHIP;
  ESKCHIP.setId(kID);
  ESKCHIP.setFiberId(fiber);
  ESKCHIP.setBC(kBC);
  ESKCHIP.setEC(kEC);
  ESKCHIP.setFlag1(kFlag1);
  ESKCHIP.setFlag2(kFlag2);

  if (crcChecker.isCrcOk()) { 
     ESKCHIP.setCRC(1);
     kchips.push_back(ESKCHIP);
  } else { 
     ESKCHIP.setCRC(0);
     kchips.push_back(ESKCHIP);
     return ; 
  }

  if (debug_) cout<<"Fiber : "<<fiber<<" BC : "<<kBC<<" EC : "<<kEC<<" KID : "<<kID<<" F1 : "<<kFlag1<<" F2 : "<<kFlag2<<" Chksum : "<<chksum<<endl;

  int col[4],ix[4],iy[4],adc[4][3];
  for (int i=0; i<3; ++i) {

    col[0] = (word[i*98+3] >> 8) & 0x00ff;
    col[1] = (word[i*98+3])      & 0x00ff;
    col[2] = (word[i*98+4] >> 8) & 0x00ff;
    col[3] = (word[i*98+4])      & 0x00ff;
    if (debug_) cout<<"Column : "<<col[0]<<" "<<col[1]<<" "<<col[2]<<" "<<col[3]<<endl;
  }

  for (int j=0; j<32; ++j) {    

    for (int i=0; i<3; ++i) {

      int row = ((kID-4) % 3);
      int column = (kID-4)/3;
      int edge = ( ((kID-4)%3) == 2)?1:0;

      adc[0][i] = (word[i*98+5+j*3] >> 4) & 0x0fff;
      if (edge == 0) {
	ix[0] = column*2+1;
	iy[0] = row*2+2;
      } else {
	ix[0] = column*2+1;
	iy[0] = row*2+1;
      }

      adc[1][i] = ((word[i*98+5+j*3] & 0x000f) << 8) ;
      adc[1][i] |= ((word[i*98+6+j*3] >> 8) & 0x00ff);  
      if (edge == 0) {
	ix[1] = column*2+1;
	iy[1] = row*2+1;
      } else {
	ix[1] = column*2+2;
	iy[1] = row*2+1;
      }

      adc[2][i] = ((word[i*98+6+j*3] & 0x00ff) << 4);
      adc[2][i] |= ((word[i*98+7+j*3] >> 12) & 0x000f);  
      ix[2] = column*2+2;
      iy[2] = row*2+2;
	
      adc[3][i] = (word[i*98+7+j*3])      & 0x0fff;
      if (edge == 0) {
	ix[3] = column*2+2;
	iy[3] = row*2+1;
      } else {
	ix[3] = column*2+1;
	iy[3] = row*2+2;
      }
     
    }
    
    for (int k=0; k<4; ++k) {

      if (iy[k]>=6) continue;

      ESDetId detId(j+1, ix[k], iy[k], 1, 1);
      ESDataFrame df(detId);
      df.setSize(3);
      
      for (int m=0; m<3; m++) df.setSample(m, adc[k][m]);
      
      digis.push_back(df);
      
      if (debug_) cout<<"Si : "<<detId.zside()<<" "<<detId.plane()<<" "<<detId.six()<<" "<<detId.siy()<<" "<<detId.strip()<<" ("<<kID<<","<<k<<") "<<df.sample(0).adc()<<" "<<df.sample(1).adc()<<" "<<df.sample(2).adc()<<endl;
      
    }
    
  }

}

string ESUnpackerCT::print(const  Word64 & word) const
{
  ostringstream str;
  str << "Word64:  " << reinterpret_cast<const bitset<64>&> (word);
  return str.str();
}

string ESUnpackerCT::print(const  Word16 & word) const
{
  ostringstream str;
  str << "Word16:  " << reinterpret_cast<const bitset<16>&> (word);
  return str.str();
}

string ESUnpackerCT::print(const  Word8 & word) const
{
  ostringstream str;
  str << "Word8:  " << reinterpret_cast<const bitset<8>&> (word);
  return str.str();
}


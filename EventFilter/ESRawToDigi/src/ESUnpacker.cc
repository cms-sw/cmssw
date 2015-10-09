#include "EventFilter/ESRawToDigi/interface/ESUnpacker.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/EcalDetId/interface/EcalDetIdCollections.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <fstream>

ESUnpacker::ESUnpacker(const edm::ParameterSet& ps) 
  : pset_(ps), fedId_(0), run_number_(0), orbit_number_(0), bx_(0), lv1_(0), trgtype_(0)
{

  debug_ = pset_.getUntrackedParameter<bool>("debugMode", false);
  lookup_ = ps.getParameter<edm::FileInPath>("LookupTable");

  m1  = ~(~Word64(0) << 1);
  m2  = ~(~Word64(0) << 2);
  m4  = ~(~Word64(0) << 4);
  m5  = ~(~Word64(0) << 5);
  m6  = ~(~Word64(0) << 6);
  m8  = ~(~Word64(0) << 8);
  m12 = ~(~Word64(0) << 12);
  m16 = ~(~Word64(0) << 16);
  m32 = ~(~Word64(0) << 32);

  // read in look-up table
  int nLines, iz, ip, ix, iy, fed, kchip, pace, bundle, fiber, optorx;
  std::ifstream file;
  file.open(lookup_.fullPath().c_str());
  if( file.is_open() ) {

    file >> nLines;

    for (int i=0; i<nLines; ++i) {
      file>> iz >> ip >> ix >> iy >> fed >> kchip >> pace >> bundle >> fiber >> optorx;

      zside_[kchip-1][pace-1] = iz;
      pl_[kchip-1][pace-1] = ip;
      x_[kchip-1][pace-1] = ix;
      y_[kchip-1][pace-1] = iy;      
    }
    
  } else {
    edm::LogWarning("Invalid Data")<<"ESUnpacker::ESUnpacker : Look up table file can not be found in "<<lookup_.fullPath().c_str();
  }

}

ESUnpacker::~ESUnpacker() {
}

void ESUnpacker::interpretRawData(int fedId, const FEDRawData & rawData, ESRawDataCollection & dccs, ESLocalRawDataCollection & kchips, ESDigiCollection & digis) {
  
  int nWords = rawData.size()/sizeof(Word64);
  if (nWords==0) return;
  int dccWords = 6;
  int head, kPACE[4], kFlag1, kFlag2, kBC, kEC, optoBC, optoEC;
  int kid = -1;

  ESDCCHeaderBlock ESDCCHeader;
  ESDCCHeader.setFedId(fedId);

  // Event header
  const Word64* header = reinterpret_cast<const Word64* >(rawData.data()); --header;
  bool moreHeaders = true;
  while (moreHeaders) {
    ++header;
    FEDHeader ESHeader( reinterpret_cast<const unsigned char*>(header) );
    if ( !ESHeader.check() ) {
      if (debug_) edm::LogWarning("Invalid Data")<<"ES : Failed header check !";
      return;
    }

    fedId_ = ESHeader.sourceID();
    lv1_   = ESHeader.lvl1ID();
    bx_    = ESHeader.bxID();

    if (debug_) {
      LogDebug("ESUnpacker")<<"[ESUnpacker]: FED Header candidate. Is header? "<< ESHeader.check();
      if (ESHeader.check())
	LogDebug("ESUnpacker") <<". BXID: "<<bx_<<" SourceID : "<<fedId_<<" L1ID: "<<lv1_;
      else LogDebug("ESUnpacker")<<" WARNING!, this is not a ES Header";
    }
    
    moreHeaders = ESHeader.moreHeaders();
  }
  if ( fedId != fedId_) {
    if (debug_) edm::LogWarning("Invalid Data")<<"Invalid ES data with source id " <<fedId_;
    ESDCCHeader.setDCCErrors(1);
    dccs.push_back(ESDCCHeader);
    return;
  }
  ESDCCHeader.setLV1(lv1_);
  ESDCCHeader.setBX(bx_);

  // Event trailer
  int slinkCRC = 1;
  const Word64* trailer = reinterpret_cast<const Word64* >(rawData.data())+(nWords-1); ++trailer;
  bool moreTrailers = true;
  while (moreTrailers) {
    --trailer;
    FEDTrailer ESTrailer(reinterpret_cast<const unsigned char*>(trailer));
    if ( !ESTrailer.check()) { 
      ++trailer; 
      if (debug_) edm::LogWarning("Invalid Data")<<"ES : Failed trailer check !";
      return;
    } 
    if ( ESTrailer.lenght() != nWords) {
      if (debug_) edm::LogWarning("Invalid Data")<<"Invalid ES data : the length is not correct !";
      ESDCCHeader.setDCCErrors(2);
      dccs.push_back(ESDCCHeader);
      return;
    }
    if ( ESTrailer.lenght() < 8) {
      if (debug_) edm::LogWarning("Invalid Data")<<"Invalid ES data : the length is not correct !";
      ESDCCHeader.setDCCErrors(3);
      dccs.push_back(ESDCCHeader);
      return;
    }
    slinkCRC = (*trailer >> 2 ) & 0x1;
    if (debug_)  {
      LogDebug("ESUnpacker")<<"[ESUnpacker]: FED Trailer candidate. Is trailer? "<<ESTrailer.check();
      if (ESTrailer.check())
	LogDebug("ESUnpacker")<<". Length of the ES event: "<<ESTrailer.lenght();
      else LogDebug("ESUnpacker")<<" WARNING!, this is not a ES Trailer";
    }

    moreTrailers = ESTrailer.moreTrailers();
  }

  if (slinkCRC != 0) {
    ESDCCHeader.setDCCErrors(101);
    dccs.push_back(ESDCCHeader);
    return;
  }

  // DCC data
  std::vector<int> FEch_status;
  int dccHeaderCount = 0;
  int dccLineCount = 0;
  int dccHead, dccLine;
  int dccCRC1_ = 0;
  int dccCRC2_ = 0;
  int dccCRC3_ = 0;
  for (const Word64* word=(header+1); word!=(header+dccWords+1); ++word) {
    if (debug_) LogDebug("ESUnpacker")<<"DCC   : "<<print(*word);
    dccHead = (*word >> 60) & m4;
    if (dccHead == 3) dccHeaderCount++;
    dccLine = (*word >> 56) & m4;
    dccLineCount++;
    if (dccLine != dccLineCount) {
      if (debug_) edm::LogWarning("Invalid Data")<<"Invalid ES data : DCC header order is not correct !";
      ESDCCHeader.setDCCErrors(4);
      dccs.push_back(ESDCCHeader);
      return; 
    }
    if (dccLineCount == 1) {
      dccCRC1_ = (*word >> 24) & m1; 
      dccCRC2_ = (*word >> 25) & m1; 
      dccCRC3_ = (*word >> 26) & m1; 
    } else if (dccLineCount == 2) {
      runtype_   = (*word >>  0) & m4;
      seqtype_   = (*word >>  4) & m4;
      dac_       = (*word >>  8) & m12; 
      gain_      = (*word >> 20) & m1;
      precision_ = (*word >> 21) & m1;
      trgtype_   = (*word >> 34) & m6;

      ESDCCHeader.setRunType(runtype_);
      ESDCCHeader.setSeqType(seqtype_);
      ESDCCHeader.setTriggerType(trgtype_);
      ESDCCHeader.setDAC(dac_);
      ESDCCHeader.setGain(gain_);
      ESDCCHeader.setPrecision(precision_);
    }
    if (dccLineCount == 3) {
      orbit_number_ = (*word >>  0) & m32;
      vminor_       = (*word >> 40) & m8;
      vmajor_       = (*word >> 48) & m8;

      ESDCCHeader.setOrbitNumber(orbit_number_);
      ESDCCHeader.setMajorVersion(vmajor_);
      ESDCCHeader.setMinorVersion(vminor_);
    }
    if (dccLineCount == 4) optoRX0_  = (*word >> 48) & m8;
    if (dccLineCount == 5) optoRX1_  = (*word >> 48) & m8;
    if (dccLineCount == 6) optoRX2_  = (*word >> 48) & m8;
    if (dccLineCount >=4) {
      for (unsigned int j=0; j<12; ++j) {
	FEch_[(dccLineCount-4)*12+j] = (*word >> (j*4)) & m4;
	FEch_status.push_back(FEch_[(dccLineCount-4)*12+j]);
      }
    }
  }
  if (vmajor_ < 4) {
    if (debug_) 
      edm::LogWarning("Invalid Data")<<"Invalid ES data format : "<<vmajor_<<" "<<vminor_;
    return;
  }
  if (dccHeaderCount != 6) {
    edm::LogWarning("Invalid Data")<<"Invalid ES data : DCC header lines are "<<dccHeaderCount;
    ESDCCHeader.setDCCErrors(5);
    dccs.push_back(ESDCCHeader);
    return;
  }
  ESDCCHeader.setOptoRX0(optoRX0_ + dccCRC1_);
  ESDCCHeader.setOptoRX1(optoRX1_ + dccCRC2_);
  ESDCCHeader.setOptoRX2(optoRX2_ + dccCRC3_);
  ESDCCHeader.setFEChannelStatus(FEch_status);
  int enableOptoRX[3] = {-1, -1, -1};
  int NenableOptoRX = 0; 
  if (optoRX0_ == 128) {
    enableOptoRX[NenableOptoRX] = 0;
    NenableOptoRX++;
  }
  if (optoRX1_ == 128) {
    enableOptoRX[NenableOptoRX] = 1;
    NenableOptoRX++;
  }
  if (optoRX2_ == 128) {
    enableOptoRX[NenableOptoRX] = 2;
  }

  // Event data
  int iopto = 0;
  int opto = -1;
  for (const Word64* word=(header+dccWords+1); word!=trailer; ++word) {
    if (debug_) LogDebug("ESUnpacker")<<"Event : "<<print(*word);

    head = (*word >> 60) & m4;

    if (head == 12) {
      if ((opto==0 && ESDCCHeader.getOptoRX0()==129) || (opto==1 && ESDCCHeader.getOptoRX1()==129) || (opto==2 && ESDCCHeader.getOptoRX2()==129)) 
	word2digi(kid, kPACE, *word, digis);
    } else if (head == 9) {
      kid      = (*word >> 2) & 0x07ff;
      kPACE[0] = (*word >> 16) & m1;
      kPACE[1] = (*word >> 17) & m1;
      kPACE[2] = (*word >> 18) & m1;
      kPACE[3] = (*word >> 19) & m1;
      kFlag2   = (*word >> 20) & m4;
      kFlag1   = (*word >> 24) & m8;
      kBC      = (*word >> 32) & m16;
      kEC      = (*word >> 48) & m8;

      ESKCHIPBlock ESKCHIP;
      ESKCHIP.setId(kid);
      ESKCHIP.setBC(kBC);
      ESKCHIP.setEC(kEC);
      ESKCHIP.setOptoBC(optoBC);
      ESKCHIP.setOptoEC(optoEC);
      ESKCHIP.setFlag1(kFlag1);
      ESKCHIP.setFlag2(kFlag2);
      kchips.push_back(ESKCHIP);
    } else if (head == 6) {
      optoBC = (*word >> 32) & m16;
      optoEC = (*word >> 48) & m8;      

      opto = enableOptoRX[iopto];
      if (opto==0) ESDCCHeader.setOptoBC0(optoBC);
      else if (opto==1) ESDCCHeader.setOptoBC1(optoBC);
      else if (opto==2) ESDCCHeader.setOptoBC2(optoBC);
      if (iopto < 2) ++iopto;
    }
  }

  dccs.push_back(ESDCCHeader);
}

void ESUnpacker::word2digi(int kid, int kPACE[4], const Word64 & word, ESDigiCollection & digis) 
{

  int pace  = (word >> 53) & m2;
  if (kPACE[pace]==0) return;
  if (kid > 1511 || kid < 1) return;
  
  int adc[3];
  adc[0]    = (word >> 0)  & m16;
  adc[1]    = (word >> 16) & m16;
  adc[2]    = (word >> 32) & m16;
  int strip = (word >> 48) & m5;

  if (debug_) LogDebug("ESUnpacker")<<kid<<" "<<strip<<" "<<pace<<" "<<adc[0]<<" "<<adc[1]<<" "<<adc[2];

  int zside, plane, ix, iy;
  zside = zside_[kid-1][pace];
  plane = pl_[kid-1][pace];
  ix    = x_[kid-1][pace];
  iy    = y_[kid-1][pace];
  
  // convert strip number from electronics id to detector id
  if (vmajor_ == 4 && (vminor_==2 || vminor_==3)) {
    if (zside == 1 && plane == 1 && iy <= 20) strip = 31 - strip;
    if (zside == 1 && plane == 2 && ix > 20) strip = 31 - strip;
    if (zside == -1 && plane == 1 && iy > 20) strip = 31 - strip;
    if (zside == -1 && plane == 2 && ix <= 20) strip = 31 - strip;
  }

  if (debug_) LogDebug("ESUnpacker")<<"DetId : "<<zside<<" "<<plane<<" "<<ix<<" "<<iy<<" "<<strip+1;
  
  if (ESDetId::validDetId(strip+1, ix, iy, plane, zside)) {
    
    ESDetId detId(strip+1, ix, iy, plane, zside);
    ESDataFrame df(detId);
    df.setSize(3);
    
    for (int i=0; i<3; i++) df.setSample(i, adc[i]);  
    
    digis.push_back(df);
    
    if (debug_) 
      LogDebug("ESUnpacker")<<"Si : "<<detId.zside()<<" "<<detId.plane()<<" "<<detId.six()<<" "<<detId.siy()<<" "<<detId.strip()<<" ("<<kid<<","<<pace<<") "<<df.sample(0).adc()<<" "<<df.sample(1).adc()<<" "<<df.sample(2).adc();
  }

}

std::string ESUnpacker::print(const  Word64 & word) const
{
  std::ostringstream str;
  str << "Word64:  " << reinterpret_cast<const std::bitset<64>&> (word);
  return str.str();
}


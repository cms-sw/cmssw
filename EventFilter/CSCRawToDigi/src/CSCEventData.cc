#include "EventFilter/CSCRawToDigi/interface/CSCEventData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCCFEBData.h"
#include "EventFilter/CSCRawToDigi/interface/cscPackerCompare.h"
#include "EventFilter/CSCRawToDigi/interface/bitset_append.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigi.h"
#include "DataFormats/CSCDigi/interface/CSCConstants.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <iostream>
#include <iterator>

#ifdef LOCAL_UNPACK
bool CSCEventData::debug = false;
#else
std::atomic<bool> CSCEventData::debug{false};
#endif

CSCEventData::CSCEventData(int chamberType, uint16_t format_version)
    : theDMBHeader(format_version),
      theALCTHeader(nullptr),
      theAnodeData(nullptr),
      theALCTTrailer(nullptr),
      theTMBData(nullptr),
      theDMBTrailer(format_version),
      theChamberType(chamberType),
      alctZSErecovered(nullptr),
      zseEnable(0),
      theFormatVersion(format_version) {
  for (unsigned i = 0; i < CSCConstants::MAX_CFEBS_RUN2; ++i) {
    theCFEBData[i] = nullptr;
  }
}

CSCEventData::CSCEventData(const uint16_t* buf, uint16_t format_version) : theFormatVersion(format_version) {
  theFormatVersion = format_version;
  unpack_data(buf);
}

void CSCEventData::unpack_data(const uint16_t* buf) {
  // zero everything
  init();
  const uint16_t* pos = buf;
  if (debug) {
    LogTrace("CSCEventData|CSCRawToDigi") << "The event data ";
    for (int i = 0; i < 16; ++i) {
      LogTrace("CSCEventData|CSCRawToDigi") << std::hex << pos[i] << " ";
    }
  }

  theDMBHeader = CSCDMBHeader(pos, theFormatVersion);
  if (!(theDMBHeader.check())) {
    LogTrace("CSCEventData|CSCRawToDigi") << "Bad DMB Header??? "
                                          << " first four words: ";
    for (int i = 0; i < 4; ++i) {
      LogTrace("CSCEventData|CSCRawToDigi") << std::hex << pos[i] << " ";
    }
  }

  if (debug) {
    LogTrace("CSCEventData|CSCRawToDigi") << "nalct = " << nalct();
    LogTrace("CSCEventData|CSCRawToDigi") << "nclct = " << nclct();
  }

  if (debug) {
    LogTrace("CSCEventData|CSCRawToDigi") << "size in words of DMBHeader" << theDMBHeader.sizeInWords();
    LogTrace("CSCEventData|CSCRawToDigi") << "sizeof(DMBHeader)" << sizeof(theDMBHeader);
  }

  pos += theDMBHeader.sizeInWords();

  if (nalct() == 1) {
    if (isALCT(pos))  //checking for ALCTData
    {
      theALCTHeader = new CSCALCTHeader(pos);
      if (!theALCTHeader->check()) {
        LogTrace("CSCEventData|CSCRawToDigi") << "+++WARNING: Corrupt ALCT data - won't attempt to decode";
      } else {
        //dataPresent|=0x40;
        pos += theALCTHeader->sizeInWords();  //size of the header
        //fill ALCT Digis
        theALCTHeader->ALCTDigis();

        //theAnodeData = new CSCAnodeData(*theALCTHeader, pos);

        /// The size of the ALCT payload is determined here
        /*
              std::cout << " ****The ALCT information from CSCEventData.cc (begin)**** " << std::endl; ///to_rm
              std::cout << " alctHeader2007().size: " << theALCTHeader->alctHeader2007().sizeInWords() << std::endl; ///to_rm
              std::cout << " ALCT Header Content: " << std::endl; ///to_rm
              /// to_rm (6 lines)
              for(int k=0; k<theALCTHeader->sizeInWords(); k+=4){
                 std::cout << std::hex << theALCTHeader->data()[k+3]
                           << " " << theALCTHeader->data()[k+2]
                           << " " << theALCTHeader->data()[k+1]
                           << " " << theALCTHeader->data()[k] << std::dec << std::endl;
                 }
               */
        //std::cout << " ALCT Size: " << theAnodeData->sizeInWords() << std::endl;
        /// Check if Zero Suppression ALCT Enabled
        // int zseEnable = 0;
        zseEnable = (theALCTHeader->data()[5] & 0x1000) >> 12;
        //std::cout << " ZSE Bit: " <<  zseEnable << std::endl; /// to_rm
        int sizeInWord_ZSE = 0;

        //alctZSErecovered = new unsigned short [theAnodeData->sizeInWords()];

        if (zseEnable) {
          /// Aauxilary variables neede to recover zero suppression
          /// Calculate the number of wire groups per layer
          int nWGs_per_layer = ((theALCTHeader->data()[6] & 0x0007) + 1) * 16;
          /// Calculate the number of words in the layer
          int nWG_round_up = int(nWGs_per_layer / 12) + (nWGs_per_layer % 3 ? 1 : 0);
          //std::cout << " Words per layer: " << nWG_round_up << std::endl; ///to_rm
          const uint16_t* posZSE = pos;
          std::vector<unsigned short> alctZSErecoveredVector;
          alctZSErecoveredVector.clear();

          //alctZSErecovered = new unsigned short [theAnodeData->sizeInWords()];
          //delete [] alctZSErecovered;
          //std::cout << " ALCT Buffer with ZSE: " << std::endl; ///to_rm
          /// unsigned short * posZSEtmpALCT = pos;
          /// This is just to dump the actual ALCT payload ** begin **
          /// For debuggin purposes
          //unsigned short * posZSEdebug = pos; ///to_rm

          /// to_rm (8 lines)
          /*
                  while (*posZSEdebug != 0xDE0D){
                        unsigned short d = *posZSEdebug;
                        unsigned short c = *(posZSEdebug+1);
                        unsigned short b = *(posZSEdebug+2);
                        unsigned short a = *(posZSEdebug+3);
                        posZSEdebug+=4;
                        std::cout << std::hex << a << " " << b << " " << c << " " << d << std::dec << std::endl;
                  }
                  */
          /// This is just to dump the actual ALCT payload ** end **

          /// Actual word counting and recovering the original ALCT payload
          int alctZSErecoveredPos = 0;
          while (*posZSE != 0xDE0D) {
            if ((*posZSE == 0x1000) && (*posZSE != 0x3000)) {
              for (int j = 0; j < nWG_round_up; j++) {
                alctZSErecoveredVector.push_back(0x0000);
              }
              alctZSErecoveredPos += nWG_round_up;
            } else {
              alctZSErecoveredVector.push_back(*posZSE);
              ++alctZSErecoveredPos;
            }
            posZSE++;
            sizeInWord_ZSE++;
          }

          alctZSErecovered = new unsigned short[alctZSErecoveredVector.size()];

          /// Convert the recovered vector into the array
          for (int l = 0; l < (int)alctZSErecoveredVector.size(); l++) {
            alctZSErecovered[l] = alctZSErecoveredVector[l];
          }

          unsigned short* posRecovered = alctZSErecovered;
          theAnodeData = new CSCAnodeData(*theALCTHeader, posRecovered);

          /// This is to check the content of the recovered ALCT payload
          /// to_rm (7 lines)
          /*
                  std::cout << " The ALCT payload recovered: " << std::endl;
                  for(int k=0; k<theAnodeData->sizeInWords(); k+=4){
                     std::cout << std::hex << alctZSErecovered[k+3] << " "
                                           << alctZSErecovered[k+2] << " "
                                           << alctZSErecovered[k+1] << " "
                                           << alctZSErecovered[k] << std::dec << std::endl;
                  }
                  */
          //delete [] alctZSErecovered;
          //std::cout << " ALCT SizeZSE : " << sizeInWord_ZSE << std::endl; ///to_rm
          //std::cout << " ALCT SizeZSE Recovered: " << alctZSErecoveredPos << std::endl; ///to_rm
          //std::cout << " ALCT Size Expected: " << theAnodeData->sizeInWords() << std::endl; ///to_rm
          pos += sizeInWord_ZSE;
        } else {
          //pos +=sizeInWord_ZSE;
          theAnodeData = new CSCAnodeData(*theALCTHeader, pos);
          pos += theAnodeData->sizeInWords();  // size of the data is determined during unpacking
        }
        //std::cout << " ****The ALCT information from CSCEventData.cc (end)**** " << std::endl; ///to_rm
        theALCTTrailer = new CSCALCTTrailer(pos);
        pos += theALCTTrailer->sizeInWords();
      }
    } else {
      LogTrace("CSCEventData|CSCRawToDigi") << "Error:nalct reported but no ALCT data found!!!";
    }
  }

  if (nclct() == 1) {
    if (isTMB(pos)) {
      //dataPresent|=0x20;
      theTMBData = new CSCTMBData(pos);  //fill all TMB data
      pos += theTMBData->size();
    } else {
      LogTrace("CSCEventData|CSCRawToDigi") << "Error:nclct reported but no TMB data found!!!";
    }
  }

  //now let's try to find and unpack the DMBTrailer
  bool dmbTrailerReached = false;
  for (int i = 0; i < 12000; ++i)  //8000 max for cfeb + 1980ALCT + 287 TMB
  {
    dmbTrailerReached = (*(i + pos) & 0xF000) == 0xF000 && (*(i + pos + 1) & 0xF000) == 0xF000 &&
                        (*(i + pos + 2) & 0xF000) == 0xF000 && (*(i + pos + 3) & 0xF000) == 0xF000 &&
                        (*(i + pos + 4) & 0xF000) == 0xE000 && (*(i + pos + 5) & 0xF000) == 0xE000 &&
                        (*(i + pos + 6) & 0xF000) == 0xE000 && (*(i + pos + 7) & 0xF000) == 0xE000;
    if (dmbTrailerReached) {
      // theDMBTrailer = *( (CSCDMBTrailer *) (pos+i) );
      theDMBTrailer = CSCDMBTrailer(pos + i, theFormatVersion);
      break;
    }
  }
  if (dmbTrailerReached) {
    for (int icfeb = 0; icfeb < CSCConstants::MAX_CFEBS_RUN2; ++icfeb) {
      theCFEBData[icfeb] = nullptr;
      int cfeb_available = theDMBHeader.cfebAvailable(icfeb);
      unsigned int cfebTimeout = theDMBTrailer.cfeb_starttimeout() | theDMBTrailer.cfeb_endtimeout();
      //cfeb_available cannot be trusted - need additional verification!
      if (cfeb_available == 1) {
        if ((cfebTimeout >> icfeb) & 1) {
          if (debug)
            LogTrace("CSCEventData|CSCRawToDigi") << "CFEB Timed out! ";
        } else {
          //dataPresent|=(0x1>>icfeb);
          // Fill CFEB data and convert it into cathode digis

          // Check if we have here DCFEB  using DMB format version field (new ME11 with DCFEBs - 0x2, other chamber types 0x1)
          bool isDCFEB = false;
          if (theDMBHeader.format_version() == 2)
            isDCFEB = true;

          theCFEBData[icfeb] = new CSCCFEBData(icfeb, pos, theFormatVersion, isDCFEB);
          pos += theCFEBData[icfeb]->sizeInWords();
        }
      }
    }
    pos += theDMBTrailer.sizeInWords();
    size_ = pos - buf;
  } else {
    LogTrace("CSCEventData|CSCRawToDigi") << "Critical Error: DMB Trailer was not found!!! ";
  }

  // std::cout << "CSC format: " << theFormatVersion << " " << getFormatVersion() << std::endl;
}

bool CSCEventData::isALCT(const short unsigned int* buf) {
  return (((buf[0] & 0xFFFF) == 0xDB0A) || (((buf[0] & 0xF800) == 0x6000) && ((buf[1] & 0xF800) == 0)));
}

bool CSCEventData::isTMB(const short unsigned int* buf) { return ((buf[0] & 0xFFF) == 0xB0C); }

CSCEventData::CSCEventData(const CSCEventData& data) { copy(data); }

CSCEventData::~CSCEventData() { destroy(); }

CSCEventData CSCEventData::operator=(const CSCEventData& data) {
  // check for self-assignment before destructing
  if (&data != this)
    destroy();
  copy(data);
  return *this;
}

void CSCEventData::init() {
  //dataPresent = 0;
  theALCTHeader = nullptr;
  theAnodeData = nullptr;
  theALCTTrailer = nullptr;
  theTMBData = nullptr;
  for (int icfeb = 0; icfeb < CSCConstants::MAX_CFEBS_RUN2; ++icfeb) {
    theCFEBData[icfeb] = nullptr;
  }
  alctZSErecovered = nullptr;
  zseEnable = 0;
}

void CSCEventData::copy(const CSCEventData& data) {
  init();
  theFormatVersion = data.theFormatVersion;
  theDMBHeader = data.theDMBHeader;
  theDMBTrailer = data.theDMBTrailer;
  if (data.theALCTHeader != nullptr)
    theALCTHeader = new CSCALCTHeader(*(data.theALCTHeader));
  if (data.theAnodeData != nullptr)
    theAnodeData = new CSCAnodeData(*(data.theAnodeData));
  if (data.theALCTTrailer != nullptr)
    theALCTTrailer = new CSCALCTTrailer(*(data.theALCTTrailer));
  if (data.theTMBData != nullptr)
    theTMBData = new CSCTMBData(*(data.theTMBData));
  for (int icfeb = 0; icfeb < CSCConstants::MAX_CFEBS_RUN2; ++icfeb) {
    theCFEBData[icfeb] = nullptr;
    if (data.theCFEBData[icfeb] != nullptr)
      theCFEBData[icfeb] = new CSCCFEBData(*(data.theCFEBData[icfeb]));
  }
  size_ = data.size_;
  theChamberType = data.theChamberType;
}

void CSCEventData::destroy() {
  if (zseEnable) {
    delete[] alctZSErecovered;
  }
  delete theALCTHeader;
  delete theAnodeData;
  delete theALCTTrailer;
  delete theTMBData;
  for (int icfeb = 0; icfeb < CSCConstants::MAX_CFEBS_RUN2; ++icfeb) {
    delete theCFEBData[icfeb];
  }
  /*
    std::cout << "Before delete alctZSErecovered " << std::endl;
    delete [] alctZSErecovered;
    std::cout << "After delete alctZSErecovered " << std::endl;
  */
}

std::vector<CSCStripDigi> CSCEventData::stripDigis(const CSCDetId& idlayer) const {
  std::vector<CSCStripDigi> result;
  for (unsigned icfeb = 0; icfeb < CSCConstants::MAX_CFEBS_RUN2; ++icfeb) {
    std::vector<CSCStripDigi> newDigis = stripDigis(idlayer, icfeb);
    result.insert(result.end(), newDigis.begin(), newDigis.end());
  }
  return result;
}

std::vector<CSCStripDigi> CSCEventData::stripDigis(unsigned idlayer, unsigned icfeb) const {
  std::vector<CSCStripDigi> result;
  if (theCFEBData[icfeb] != nullptr) {
    std::vector<CSCStripDigi> newDigis = theCFEBData[icfeb]->digis(idlayer);
    result.insert(result.end(), newDigis.begin(), newDigis.end());
  }

  return result;
}

std::vector<CSCWireDigi> CSCEventData::wireDigis(unsigned ilayer) const {
  if (theAnodeData == nullptr) {
    return std::vector<CSCWireDigi>();
  } else {
    return theAnodeData->wireDigis(ilayer);
  }
}

std::vector<std::vector<CSCStripDigi> > CSCEventData::stripDigis() const {
  std::vector<std::vector<CSCStripDigi> > result;
  for (int layer = CSCDetId::minLayerId(); layer <= CSCDetId::maxLayerId(); ++layer) {
    std::vector<CSCStripDigi> digis = stripDigis(layer);
    result.push_back(digis);
  }
  return result;
}

std::vector<std::vector<CSCWireDigi> > CSCEventData::wireDigis() const {
  std::vector<std::vector<CSCWireDigi> > result;
  for (int layer = CSCDetId::minLayerId(); layer <= CSCDetId::maxLayerId(); ++layer) {
    result.push_back(wireDigis(layer));
  }
  return result;
}

const CSCCFEBData* CSCEventData::cfebData(unsigned icfeb) const { return theCFEBData[icfeb]; }

CSCALCTHeader* CSCEventData::alctHeader() const {
  if (nalct() == 0)
    throw cms::Exception("No ALCT for this chamber");
  return theALCTHeader;
}

CSCALCTTrailer* CSCEventData::alctTrailer() const {
  if (nalct() == 0)
    throw cms::Exception("No ALCT for this chamber");
  return theALCTTrailer;
}

CSCAnodeData* CSCEventData::alctData() const {
  if (nalct() == 0)
    throw cms::Exception("No ALCT for this chamber");
  return theAnodeData;
}

CSCTMBData* CSCEventData::tmbData() const {
  if (nclct() == 0)
    throw cms::Exception("No CLCT for this chamber");
  return theTMBData;
}

CSCTMBHeader* CSCEventData::tmbHeader() const {
  if ((nclct() == 0) || (tmbData() == nullptr))
    throw cms::Exception("No CLCT header for this chamber");
  return tmbData()->tmbHeader();
}

CSCComparatorData* CSCEventData::comparatorData() const {
  if ((nclct() == 0) || (tmbData() == nullptr))
    throw cms::Exception("No CLCT data for this chamber");
  return tmbData()->comparatorData();
}

void CSCEventData::setEventInformation(int bxnum, int lvl1num) {
  theDMBHeader.setBXN(bxnum);
  theDMBHeader.setL1A(lvl1num);
  theDMBHeader.setL1A24(lvl1num);
  if (theALCTHeader) {
    theALCTHeader->setEventInformation(theDMBHeader);
  }
  if (theTMBData) {
    theTMBData->tmbHeader()->setEventInformation(theDMBHeader);

    assert(theChamberType > 0);

    theTMBData->tmbHeader()->setNCFEBs(CSCConstants::MAX_CFEBS_RUN1);

    // Set number of CFEBs to 7 for Post-LS1 ME11
    if ((theFormatVersion >= 2013) && ((theChamberType == 1) || (theChamberType == 2))) {
      theTMBData->tmbHeader()->setNCFEBs(CSCConstants::MAX_CFEBS_RUN2);
    }
  }
  for (unsigned cfeb = 0; cfeb < CSCConstants::MAX_CFEBS_RUN2; cfeb++) {
    if (theCFEBData[cfeb])
      theCFEBData[cfeb]->setL1A(lvl1num);
  }
}

void CSCEventData::checkALCTClasses() {
  if (theAnodeData == nullptr) {
    assert(theChamberType > 0);
    theALCTHeader = new CSCALCTHeader(theChamberType);
    theALCTHeader->setEventInformation(theDMBHeader);
    theAnodeData = new CSCAnodeData(*theALCTHeader);
    int size = theALCTHeader->sizeInWords() + theAnodeData->sizeInWords() + CSCALCTTrailer::sizeInWords();
    theALCTTrailer = new CSCALCTTrailer(size, theALCTHeader->alctFirmwareVersion());
    // set data available flag
    theDMBHeader.addNALCT();
  }
}

void CSCEventData::checkTMBClasses() {
  int nCFEBs = CSCConstants::MAX_CFEBS_RUN1;
  if ((theFormatVersion >= 2013) && ((theChamberType == 1) || (theChamberType == 2))) {
    nCFEBs = CSCConstants::MAX_CFEBS_RUN2;
  }
  if (theTMBData == nullptr) {
    if (theFormatVersion == 2013) {  // Set to TMB format for Post-LS1/Run2 data
      theTMBData = new CSCTMBData(2013, 0x7a76, nCFEBs);
    } else if (theFormatVersion == 2020) {  // Set to TMB format for Run3 data
      if ((theChamberType == 1) || (theChamberType == 2)) {
        theTMBData = new CSCTMBData(2020, 0x602, nCFEBs);  // ME11 GEM fw
      } else {
        theTMBData = new CSCTMBData(2020, 0x403);  // MEx1 CCLUT fw
      }
    } else {
      theTMBData = new CSCTMBData(2007, 0x50c3);
    }
    theTMBData->tmbHeader()->setEventInformation(theDMBHeader);
    theDMBHeader.addNCLCT();
  }
  theTMBData->tmbHeader()->setNCFEBs(nCFEBs);
}

void CSCEventData::add(const CSCStripDigi& digi, int layer) {
  //@@ need special logic here for ME11
  unsigned cfeb = digi.getCFEB();
  bool sixteenSamples = false;
  if (digi.getADCCounts().size() == 16)
    sixteenSamples = true;
  if (theCFEBData[cfeb] == nullptr) {
    bool isDCFEB = false;
    if (theDMBHeader.format_version() == 2)
      isDCFEB = true;
    theCFEBData[cfeb] = new CSCCFEBData(cfeb, sixteenSamples, theFormatVersion, isDCFEB);
    theDMBHeader.addCFEB(cfeb);
  }
  theCFEBData[cfeb]->add(digi, layer);
}

void CSCEventData::add(const CSCWireDigi& digi, int layer) {
  checkALCTClasses();
  theAnodeData->add(digi, layer);
  theALCTHeader->setDAVForChannel(digi.getWireGroup());
  theALCTHeader->setBXNCount(digi.getWireGroupBX());
}

void CSCEventData::add(const CSCComparatorDigi& digi, int layer) {
  checkTMBClasses();
  theTMBData->comparatorData()->add(digi, layer);
}

void CSCEventData::add(const CSCComparatorDigi& digi, const CSCDetId& cid) {
  checkTMBClasses();
  theTMBData->comparatorData()->add(digi, cid);
}

void CSCEventData::add(const std::vector<CSCALCTDigi>& digis) {
  checkALCTClasses();
  theALCTHeader->add(digis);
}

void CSCEventData::add(const std::vector<CSCCLCTDigi>& digis) {
  checkTMBClasses();
  theTMBData->tmbHeader()->add(digis);
}

void CSCEventData::add(const std::vector<CSCCorrelatedLCTDigi>& digis) {
  checkTMBClasses();
  theTMBData->tmbHeader()->add(digis);
}

/// Add/pack LCT CSCShower object
void CSCEventData::addShower(const std::vector<CSCShowerDigi>& digis) {
  checkTMBClasses();
  for (auto it : digis) {
    theTMBData->tmbHeader()->addShower(it);
  }
}

/// Add/pack anode CSCShower object (from OTMB header)
void CSCEventData::addAnodeShower(const std::vector<CSCShowerDigi>& digis) {
  checkTMBClasses();
  for (auto it : digis) {
    theTMBData->tmbHeader()->addAnodeShower(it);
  }
}

/// Add/pack cathode CSCShower object (from OTMB header)
void CSCEventData::addCathodeShower(const std::vector<CSCShowerDigi>& digis) {
  checkTMBClasses();
  for (auto it : digis) {
    theTMBData->tmbHeader()->addCathodeShower(it);
  }
}

/// Add/pack anode CSCShower objects (from ALCT board data)
void CSCEventData::addAnodeALCTShower(const std::vector<CSCShowerDigi>& digis) {
  checkALCTClasses();
  theALCTHeader->addShower(digis);
}

/// Add/pack GE11 GEM Pad Clusters trigger objects received by OTMB from GEM
void CSCEventData::add(const std::vector<GEMPadDigiCluster>& clusters, const GEMDetId& gemdetid) {
  checkTMBClasses();
  if (theTMBData->hasGEM()) {
    int gem_layer = gemdetid.layer();
    int eta_roll = gemdetid.roll();
    for (const auto& it : clusters) {
      if (it.isValid())
        theTMBData->tmbHeader()->setALCTMatchTime(it.alctMatchTime());
      theTMBData->gemData()->addEtaPadCluster(it, gem_layer - 1, 8 - eta_roll);
    }
  }
}

std::ostream& operator<<(std::ostream& os, const CSCEventData& evt) {
  for (int ilayer = CSCDetId::minLayerId(); ilayer <= CSCDetId::maxLayerId(); ++ilayer) {
    std::vector<CSCStripDigi> stripDigis = evt.stripDigis(ilayer);
    //copy(stripDigis.begin(), stripDigis.end(), std::ostream_iterator<CSCStripDigi>(os, "\n"));
    //print your scas here
    std::vector<CSCWireDigi> wireDigis = evt.wireDigis(ilayer);
    //copy(wireDigis.begin(), wireDigis.end(), std::ostream_iterator<CSCWireDigi>(os, "\n"));
  }
  return os;
}

boost::dynamic_bitset<> CSCEventData::pack() {
  boost::dynamic_bitset<> result =
      bitset_utilities::ushortToBitset(theDMBHeader.sizeInWords() * 16, theDMBHeader.data());

  // Container for CRC calculations
  std::vector<std::pair<unsigned int, unsigned short*> > crcvec;

  if (theALCTHeader != nullptr) {
    boost::dynamic_bitset<> alctHeader = theALCTHeader->pack();
    result = bitset_utilities::append(result, alctHeader);
    crcvec.push_back(std::make_pair(theALCTHeader->sizeInWords(), theALCTHeader->data()));
  }
  if (theAnodeData != nullptr) {
    boost::dynamic_bitset<> anodeData =
        bitset_utilities::ushortToBitset(theAnodeData->sizeInWords() * 16, theAnodeData->data());
    result = bitset_utilities::append(result, anodeData);
    crcvec.push_back(std::make_pair(theAnodeData->sizeInWords(), theAnodeData->data()));
  }
  if (theALCTTrailer != nullptr) {
    unsigned int crc = calcALCTcrc(crcvec);
    theALCTTrailer->setCRC(crc);
    boost::dynamic_bitset<> alctTrailer =
        bitset_utilities::ushortToBitset(theALCTTrailer->sizeInWords() * 16, theALCTTrailer->data());
    result = bitset_utilities::append(result, alctTrailer);
  }
  if (theTMBData != nullptr) {
    result = bitset_utilities::append(result, theTMBData->pack());
  }

  for (int icfeb = 0; icfeb < CSCConstants::MAX_CFEBS_RUN2; ++icfeb) {
    if (theCFEBData[icfeb] != nullptr) {
      boost::dynamic_bitset<> cfebData =
          bitset_utilities::ushortToBitset(theCFEBData[icfeb]->sizeInWords() * 16, theCFEBData[icfeb]->data());
      result = bitset_utilities::append(result, cfebData);
    }
  }

  boost::dynamic_bitset<> dmbTrailer =
      bitset_utilities::ushortToBitset(theDMBTrailer.sizeInWords() * 16, theDMBTrailer.data());
  result = bitset_utilities::append(result, dmbTrailer);
  return result;
}

unsigned int CSCEventData::calcALCTcrc(std::vector<std::pair<unsigned int, unsigned short*> >& vec) {
  int CRC = 0;

  for (unsigned int n = 0; n < vec.size(); n++) {
    for (uint16_t j = 0, w = 0; j < vec[n].first; j++) {
      if (vec[n].second != nullptr) {
        w = vec[n].second[j] & 0xffff;
        for (uint32_t i = 15, t = 0, ncrc = 0; i < 16; i--) {
          t = ((w >> i) & 1) ^ ((CRC >> 21) & 1);
          ncrc = (CRC << 1) & 0x3ffffc;
          ncrc |= (t ^ (CRC & 1)) << 1;
          ncrc |= t;
          CRC = ncrc;
        }
      }
    }
  }

  return CRC;
}

void CSCEventData::selfTest() {
  CSCEventData chamberData(5);
  CSCDetId detId(1, 3, 2, 1, 3);
  std::vector<CSCCLCTDigi> clctDigis;
  // Both CLCTs are read-out at the same (pre-trigger) bx, so the last-but-one
  // arguments in both digis must be the same.
  clctDigis.push_back(CSCCLCTDigi(1, 1, 4, 1, 0, 30, 3, 2, 1));  // valid for 2007
  clctDigis.push_back(CSCCLCTDigi(1, 1, 2, 1, 1, 31, 1, 2, 2));

  // BX of LCT (8th argument) is 1-bit word (the least-significant bit
  // of ALCT's bx).
  std::vector<CSCCorrelatedLCTDigi> corrDigis;
  corrDigis.push_back(CSCCorrelatedLCTDigi(1, 1, 2, 10, 98, 5, 0, 1, 0, 0, 0, 0));
  corrDigis.push_back(CSCCorrelatedLCTDigi(2, 1, 2, 20, 15, 9, 1, 0, 0, 0, 0, 0));

  chamberData.add(clctDigis);
  chamberData.add(corrDigis);

  CSCWireDigi wireDigi(10, 6);
  CSCComparatorDigi comparatorDigi(30, 1, 6);
  chamberData.add(wireDigi, 3);
  chamberData.add(comparatorDigi, 3);

  CSCEventData newData = cscPackAndUnpack(chamberData);

  std::vector<CSCCLCTDigi> clcts = newData.tmbHeader()->CLCTDigis(detId.rawId());
  assert(cscPackerCompare(clcts[0], clctDigis[0]));
  assert(cscPackerCompare(clcts[1], clctDigis[1]));

  std::vector<CSCCorrelatedLCTDigi> lcts = newData.tmbHeader()->CorrelatedLCTDigis(detId.rawId());
  assert(cscPackerCompare(lcts[0], corrDigis[0]));
  assert(cscPackerCompare(lcts[1], corrDigis[1]));

  // test strip digis
  CSCDetId me1adet1(1, 1, 1, 4, 1);
  CSCDetId me1bdet1(1, 1, 4, 4, 6);
  CSCDetId me1adet2(2, 1, 1, 4, 2);
  CSCDetId me1bdet2(2, 1, 4, 4, 5);

  std::vector<int> sca(16, 600);
  std::vector<unsigned short> overflow(16, 0), overlap(16, 0), errorfl(16, 0);
  CSCStripDigi me1a(5, sca, overflow, overlap, errorfl);
  CSCStripDigi me1b(8, sca, overflow, overlap, errorfl);

  CSCEventData forward(1);
  CSCEventData backward(1);

  forward.add(me1a, me1adet1.layer());
  forward.add(me1b, me1bdet1.layer());
  backward.add(me1a, me1adet2.layer());
  backward.add(me1b, me1adet2.layer());
  std::vector<CSCStripDigi> me1afs = forward.stripDigis(me1adet1);
  std::vector<CSCStripDigi> me1bfs = forward.stripDigis(me1bdet1);
  std::vector<CSCStripDigi> me1abs = backward.stripDigis(me1adet2);
  std::vector<CSCStripDigi> me1bbs = backward.stripDigis(me1bdet2);
  //FIXME The current code works under the assumption that ME11 and ME1A
  // go into separate EventData.  They need to be combined.
  assert(me1afs.size() == 16);
  assert(me1bfs.size() == 16);
  assert(me1abs.size() == 16);
  assert(me1bbs.size() == 16);

  assert(me1afs[4].getStrip() == 5);
  assert(me1bfs[7].getStrip() == 8);
  assert(me1abs[4].getStrip() == 5);
  assert(me1bbs[7].getStrip() == 8);
  assert(me1afs[4].pedestal() == 600);
  assert(me1bfs[7].pedestal() == 600);
  assert(me1abs[4].pedestal() == 600);
  assert(me1bbs[7].pedestal() == 600);
}

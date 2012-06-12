#include "EventFilter/SiPixelRawToDigi/interface/PixelDataFormatter.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingTree.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFrameConverter.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"

#include "CondFormats/SiPixelObjects/interface/PixelROC.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <bitset>
#include <sstream>
#include <iostream>

using namespace std;
using namespace edm;
using namespace sipixelobjects;

const int LINK_bits = 6;
const int ROC_bits  = 5;
const int DCOL_bits = 5;
const int PXID_bits = 8;
const int ADC_bits  = 8;

const int ADC_shift  = 0;
const int PXID_shift = ADC_shift + ADC_bits;
const int DCOL_shift = PXID_shift + PXID_bits;
const int ROC_shift  = DCOL_shift + DCOL_bits;
const int LINK_shift = ROC_shift + ROC_bits;

const PixelDataFormatter::Word32 LINK_mask = ~(~PixelDataFormatter::Word32(0) << LINK_bits);
const PixelDataFormatter::Word32 ROC_mask  = ~(~PixelDataFormatter::Word32(0) << ROC_bits);
const PixelDataFormatter::Word32 DCOL_mask = ~(~PixelDataFormatter::Word32(0) << DCOL_bits);
const PixelDataFormatter::Word32 PXID_mask = ~(~PixelDataFormatter::Word32(0) << PXID_bits);
const PixelDataFormatter::Word32 ADC_mask  = ~(~PixelDataFormatter::Word32(0) << ADC_bits);

const PixelDataFormatter::Word64 WORD32_mask = 0xffffffff;


PixelDataFormatter::PixelDataFormatter( const SiPixelFedCabling* map)
  : theDigiCounter(0), theWordCounter(0), theCablingTree(map), badPixelInfo(0), modulesToUnpack(0)
{
  int s32 = sizeof(Word32);
  int s64 = sizeof(Word64);
  int s8  = sizeof(char);
  if ( s8 != 1 || s32 != 4*s8 || s64 != 2*s32) {
     LogError("**PixelDataFormatter**")
          <<" unexpected sizes: "
          <<"  size of char is: " << s8
          <<", size of Word32 is: " << s32
          <<", size of Word64 is: " << s64
          <<", send exception" ;
  }
  includeErrors = false;
  useQualityInfo = false;
  allDetDigis = 0;
  hasDetDigis = 0;
}

void PixelDataFormatter::setErrorStatus(bool ErrorStatus)
{
  includeErrors = ErrorStatus;
  errorcheck.setErrorStatus(includeErrors);
}

void PixelDataFormatter::setQualityStatus(bool QualityStatus, const SiPixelQuality* QualityInfo)
{
  useQualityInfo = QualityStatus;
  badPixelInfo = QualityInfo;
}

void PixelDataFormatter::setModulesToUnpack(const std::set<unsigned int> * moduleIds)
{
  modulesToUnpack = moduleIds;
}

void PixelDataFormatter::passFrameReverter(const SiPixelFrameReverter* reverter)
{
  theFrameReverter = reverter;
}

void PixelDataFormatter::interpretRawData(bool& errorsInEvent, int fedId, const FEDRawData& rawData, Digis& digis, Errors& errors)
{
  debug = edm::MessageDrop::instance()->debugEnabled;
  int nWords = rawData.size()/sizeof(Word64);
  if (nWords==0) return;

  SiPixelFrameConverter * converter = (theCablingTree) ? 
      new SiPixelFrameConverter(theCablingTree, fedId) : 0;

  // check CRC bit
  const Word64* trailer = reinterpret_cast<const Word64* >(rawData.data())+(nWords-1);
  bool CRC_OK = errorcheck.checkCRC(errorsInEvent, fedId, trailer, errors);
  
  if(CRC_OK) {
    // check headers
    const Word64* header = reinterpret_cast<const Word64* >(rawData.data()); header--;
    bool moreHeaders = true;
    while (moreHeaders) {
      header++;
      if (debug) LogTrace("")<<"HEADER:  " <<  print(*header);
      bool headerStatus = errorcheck.checkHeader(errorsInEvent, fedId, header, errors);
      moreHeaders = headerStatus;
    }

    // check trailers
    bool moreTrailers = true;
    trailer++;
    while (moreTrailers) {
      trailer--;
      if (debug) LogTrace("")<<"TRAILER: " <<  print(*trailer);
      bool trailerStatus = errorcheck.checkTrailer(errorsInEvent, fedId, nWords, trailer, errors);
      moreTrailers = trailerStatus;
    }

    // data words
    theWordCounter += 2*(nWords-2);
    if (debug) LogTrace("")<<"data words: "<< (trailer-header-1);
    for (const Word64* word = header+1; word != trailer; word++) {
      if (debug) LogTrace("")<<"DATA:    " <<  print(*word);
      Word32 w1 =  *word       & WORD32_mask;
      Word32 w2 =  *word >> 32 & WORD32_mask;
      if (w1==0) theWordCounter--;
      if (w2==0) theWordCounter--;

      // check status of word...
      bool notErrorROC1 = errorcheck.checkROC(errorsInEvent, fedId, converter, w1, errors);
      if (notErrorROC1) {
        int status1 = word2digi(fedId, converter, includeErrors, useQualityInfo, w1, digis);
        if (status1) {
	    LogDebug("PixelDataFormatter::interpretRawData") 
                    << "status #" <<status1<<" returned for word1";
            errorsInEvent = true;
	    errorcheck.conversionError(fedId, converter, status1, w1, errors);
        }
      }
      bool notErrorROC2 = errorcheck.checkROC(errorsInEvent, fedId, converter, w2, errors);
      if (notErrorROC2) {
        int status2 = word2digi(fedId, converter, includeErrors, useQualityInfo, w2, digis);
        if (status2) {
	    LogDebug("PixelDataFormatter::interpretRawData") 
                    << "status #" <<status2<<" returned for word2";
            errorsInEvent = true;
	    errorcheck.conversionError(fedId, converter, status2, w2, errors);
        }
      }
    }
  }  // end if(CRC_OK)
  delete converter;
}


void PixelDataFormatter::formatRawData(unsigned int lvl1_ID, RawData & fedRawData, const Digis & digis) 
{
  std::map<int, vector<Word32> > words;

  // translate digis into 32-bit raw words and store in map indexed by Fed
  for (Digis::const_iterator im = digis.begin(); im != digis.end(); im++) {
    allDetDigis++;
    cms_uint32_t rawId = im->first;
    hasDetDigis++;
    const DetDigis & detDigis = im->second;
    for (DetDigis::const_iterator it = detDigis.begin(); it != detDigis.end(); it++) {
      theDigiCounter++;
      const PixelDigi & digi = (*it);
      int status = digi2word( rawId, digi, words);
      if (status) {
         LogError("PixelDataFormatter::formatData exception")
            <<" digi2word returns error #"<<status
            <<" Ndigis: "<<theDigiCounter << endl
            <<" detector: "<<rawId<< endl
            << print(digi) <<endl;
      } // if (status)
    } // for (DetDigis
  } // for (Digis
  LogTrace(" allDetDigis/hasDetDigis : ") << allDetDigis<<"/"<<hasDetDigis;

  typedef std::map<int, vector<Word32> >::const_iterator RI;
  for (RI feddata = words.begin(); feddata != words.end(); feddata++) {
    int fedId = feddata->first;
    // since raw words are written in the form of 64-bit packets
    // add extra 32-bit word to make number of words even if necessary
    if (words.find(fedId)->second.size() %2 != 0) words[fedId].push_back( Word32(0) );

    // size in Bytes; create output structure
    int dataSize = words.find(fedId)->second.size() * sizeof(Word32);
    int nHeaders = 1;
    int nTrailers = 1;
    dataSize += (nHeaders+nTrailers)*sizeof(Word64); 
    FEDRawData * rawData = new FEDRawData(dataSize);

    // get begining of data;
    Word64 * word = reinterpret_cast<Word64* >(rawData->data());

    // write one header
    FEDHeader::set(  reinterpret_cast<unsigned char*>(word), 0, lvl1_ID, 0, fedId); 
    word++;

    // write data
    unsigned int nWord32InFed = words.find(fedId)->second.size();
    for (unsigned int i=0; i < nWord32InFed; i+=2) {
      *word = (Word64(words.find(fedId)->second[i]) << 32 ) | words.find(fedId)->second[i+1];
      LogDebug("PixelDataFormatter")  << print(*word);
      word++;
    }

    // write one trailer
    FEDTrailer::set(  reinterpret_cast<unsigned char*>(word), dataSize/sizeof(Word64), 0,0,0);
    word++;

    // check memory
    if (word != reinterpret_cast<Word64* >(rawData->data()+dataSize)) {
      string s = "** PROBLEM in PixelDataFormatter !!!";
      throw cms::Exception(s);
    } // if (word !=
    fedRawData[fedId] = *rawData;
    delete rawData;
  } // for (RI feddata 
}

int PixelDataFormatter::digi2word( cms_uint32_t detId, const PixelDigi& digi, 
    std::map<int, vector<Word32> > & words) const
{
  LogDebug("PixelDataFormatter")
// <<" detId: " << detId 
  <<print(digi);

  DetectorIndex detector = {detId, digi.row(), digi.column()};
  ElectronicIndex cabling;
  int fedId  = theFrameReverter->toCabling(cabling, detector);
  if (fedId<0) return fedId;

  Word32 word =
             (cabling.link << LINK_shift)
           | (cabling.roc << ROC_shift)
           | (cabling.dcol << DCOL_shift)
           | (cabling.pxid << PXID_shift)
           | (digi.adc() << ADC_shift);
  words[fedId].push_back(word);
  theWordCounter++;

  return 0;
}


int PixelDataFormatter::word2digi(const int fedId, const SiPixelFrameConverter* converter, 
    const bool includeErrors, const bool useQuality, const Word32 & word, Digis & digis) const
{
  // do not interpret false digis
  if (word == 0 ) return 0;

  ElectronicIndex cabling;
  cabling.dcol = (word >> DCOL_shift) & DCOL_mask;
  cabling.pxid = (word >> PXID_shift) & PXID_mask;
  cabling.link = (word >> LINK_shift) & LINK_mask;
  cabling.roc  = (word >> ROC_shift) & ROC_mask;
  int adc      = (word >> ADC_shift) & ADC_mask;

  if (debug) {
    LocalPixel::DcolPxid pixel = {cabling.dcol,cabling.pxid};
    LocalPixel local(pixel);
    LogTrace("")<<" link: "<<cabling.link<<", roc: "<<cabling.roc
                <<" rocRow: "<<local.rocRow()<<", rocCol:"<<local.rocCol()
                <<" (dcol: "<<cabling.dcol<<", pxid:"<<cabling.pxid<<"), adc:"<<adc;
  }

  if (!converter) return 0;

  DetectorIndex detIdx;
  int status = converter->toDetector(cabling, detIdx);
  if (status) return status;

  // exclude ROC(raw) based on bad ROC list bad in SiPixelQuality
  // enable: process.siPixelDigis.UseQualityInfo = True
  // 20-10-2010 A.Y.
  if (useQuality&&badPixelInfo) {
    CablingPathToDetUnit path = {static_cast<unsigned int>(fedId),
                                 static_cast<unsigned int>(cabling.link),
                                 static_cast<unsigned int>(cabling.roc)};
    const PixelROC * roc = theCablingTree->findItem(path);
    short rocInDet = (short) roc->idInDetUnit();
    bool badROC = badPixelInfo->IsRocBad(detIdx.rawId, rocInDet);
    if (badROC) return 0;
  }

  if (modulesToUnpack && modulesToUnpack->find(detIdx.rawId) == modulesToUnpack->end()) return 0;

  PixelDigi pd(detIdx.row, detIdx.col, adc);
  digis[detIdx.rawId].push_back(pd);

  theDigiCounter++;

  if (debug)  LogTrace("") << print(pd);
  return 0;
}

std::string PixelDataFormatter::print(const PixelDigi & digi) const
{
  ostringstream str;
  str << " DIGI: row: " << digi.row() <<", col: " << digi.column() <<", adc: " << digi.adc();
  return str.str();
}

std::string PixelDataFormatter::print(const  Word64 & word) const
{
  ostringstream str;
  str  <<"word64:  " << reinterpret_cast<const bitset<64>&> (word);
  return str.str();
}


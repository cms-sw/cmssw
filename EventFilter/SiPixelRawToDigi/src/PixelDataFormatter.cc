#include "EventFilter/SiPixelRawToDigi/interface/PixelDataFormatter.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFrameConverter.h"

#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"


#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <bitset>
#include <sstream>
#include <iostream>

using namespace std;
using namespace edm;

const int PixelDataFormatter::LINK_bits = 6;
const int PixelDataFormatter::ROC_bits  = 5;
const int PixelDataFormatter::DCOL_bits = 5;
const int PixelDataFormatter::PXID_bits = 8;
const int PixelDataFormatter::ADC_bits  = 8;

const int PixelDataFormatter::ADC_shift  = 0;
const int PixelDataFormatter::PXID_shift = ADC_shift + ADC_bits;
const int PixelDataFormatter::DCOL_shift = PXID_shift + PXID_bits;
const int PixelDataFormatter::ROC_shift  = DCOL_shift + DCOL_bits;
const int PixelDataFormatter::LINK_shift = ROC_shift + ROC_bits;


PixelDataFormatter::PixelDataFormatter( const SiPixelFedCablingMap * map)
  : theDigiCounter(0), theWordCounter(0), theCablingMap(map)
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
}

void PixelDataFormatter::interpretRawData(int fedId, const FEDRawData& rawData, Digis& digis)
{
  SiPixelFrameConverter converter(theCablingMap, fedId); 
  try {
    int nWords = rawData.size()/sizeof(Word64);
    if (nWords==0) return;

    // check headers
    const Word64* header = reinterpret_cast<const Word64* >(rawData.data()); header--;
    bool moreHeaders = true;
    while (moreHeaders) {
      header++;
      FEDHeader fedHeader( reinterpret_cast<const unsigned char*>(header));
      if ( !fedHeader.check() ) break; // throw exception?
      if ( fedHeader.sourceID() != fedId)  throw cms::Exception("PROBLEM in PixelDataFormatter !");
      moreHeaders = fedHeader.moreHeaders();
    }

    // check trailers
    const Word64* trailer = reinterpret_cast<const Word64* >(rawData.data())+(nWords-1); trailer++;
    bool moreTrailers = true;
    while (moreTrailers) {
      trailer--;
      FEDTrailer fedTrailer(reinterpret_cast<const unsigned char*>(trailer)); 
      if ( !fedTrailer.check()) { trailer++; break; } // throw exception?
      if ( fedTrailer.lenght()!= nWords) {
        throw cms::Exception("PROBLEM in PixelDataFormatter !!");
      }
      moreTrailers = fedTrailer.moreTrailers();
    }

    // data words
    theWordCounter += 2*(nWords-2);
    //LogTrace("")<<"data words: "<< (trailer-header-1);
    for (const Word64* word = header+1; word != trailer; word++) {
      LogTrace("PixelDataFormatter") << print(*word);
      static const Word64 WORD32_mask  = 0xffffffff;
      Word32 w1 =  *word >> 32 & WORD32_mask;
      Word32 w2 =  *word       & WORD32_mask;
      if (w2==0) theWordCounter--;
      word2digi(converter, w1, digis);
      word2digi(converter, w2, digis);
    } 
  }
  catch ( cms::Exception & err) { LogError("PixelDataFormatter, exception") <<err.what(); }
}


FEDRawData * PixelDataFormatter::formatData(int fedId, const Digis & digis) 
{
  vector<Word32> words;

  static int allDetDigis = 0;
  static int hasDetDigis = 0;
  SiPixelFrameConverter converter(theCablingMap, fedId);
  for (Digis::const_iterator im = digis.begin(); im != digis.end(); im++) {
    allDetDigis++;
    uint32_t rawId = im->id;
//    uint32_t rawId = im->first;
    if ( !converter.hasDetUnit(rawId) ) continue;
    hasDetDigis++;
    const DetDigis & detDigis = im->data;
//    const DetDigis & detDigis = im->second;
    for (DetDigis::const_iterator it = detDigis.begin(); it != detDigis.end(); it++) {
      theDigiCounter++;
      const PixelDigi & digi = (*it);
      try { digi2word( converter, rawId, digi, words); }
      catch ( cms::Exception& e) { 
         LogError("PixelDataFormatter::formatData exception") <<e.what() 
            <<" Ndigis: "<<theDigiCounter << endl
            <<" detector: "<<rawId<< endl
            << print(digi) <<endl; 
      }
    }
  }
  LogTrace(" allDetDigis/hasDetDigis : ") << allDetDigis<<"/"<<hasDetDigis;

  //
  // since digis are writted in the form og 64-bit packets
  // add extra 32-bit word to make number of digis even
  //
  if (words.size() %2 != 0) words.push_back( Word32(0) );


  //
  // size in Bytes; create output structure
  //
  int dataSize = words.size() * sizeof(Word32);
  int nHeaders = 1;
  int nTrailers = 1;
  dataSize += (nHeaders+nTrailers)*sizeof(Word64); 
  FEDRawData * rawData = new FEDRawData(dataSize);

  //
  // get begining of data;
  Word64 * word = reinterpret_cast<Word64* >(rawData->data());

  //
  // write one header
  FEDHeader::set(  reinterpret_cast<unsigned char*>(word), 0, 0, 0, fedId); 
  word++;

  //
  // write data
  for (unsigned int i=0; i < words.size(); i+=2) {
    *word = (Word64(words[i]) << 32 ) | words[i+1];
    LogDebug("PixelDataFormatter")  << print(*word);
    word++;
  }

  // write one trailer
  FEDTrailer::set(  reinterpret_cast<unsigned char*>(word), dataSize/sizeof(Word64), 0,0,0);
  word++;

  //
  // check memory
  //
  if (word != reinterpret_cast<Word64* >(rawData->data()+dataSize)) {
    string s = "** PROBLEM in PixelDataFormatter !!!";
    throw cms::Exception(s);
  }

  return rawData;
}


void PixelDataFormatter::digi2word( const SiPixelFrameConverter& converter,
    uint32_t detId, const PixelDigi& digi, std::vector<Word32> & words) const
{
  LogDebug("PixelDataFormatter")<< print(digi);

  SiPixelFrameConverter::DetectorIndex detector = {detId, digi.row(), digi.column()};
  SiPixelFrameConverter::CablingIndex  cabling = converter.toCabling(detector);

  Word32 word =
             (cabling.link  << LINK_shift)
           | (cabling.roc << ROC_shift)
           | (cabling.dcol << DCOL_shift)
           | (cabling.pxid << PXID_shift)
           | (digi.adc() << ADC_shift);
  words.push_back(word);
  theWordCounter++;
}


void PixelDataFormatter::word2digi(const SiPixelFrameConverter& converter, 
    const Word32 & word, Digis & digis) const
{
  // do not interpret false digis
  if (word == 0 ) return;

  static const Word32 LINK_mask = ~(~Word32(0) << LINK_bits);
  static const Word32 ROC_mask  = ~(~Word32(0) << ROC_bits);
  static const Word32 DCOL_mask = ~(~Word32(0) << DCOL_bits);
  static const Word32 PXID_mask = ~(~Word32(0) << PXID_bits);
  static const Word32 ADC_mask  = ~(~Word32(0) << ADC_bits);

  SiPixelFrameConverter::CablingIndex cabling;
  cabling.dcol = (word >> DCOL_shift) & DCOL_mask;
  cabling.pxid = (word >> PXID_shift) & PXID_mask;
  cabling.link = (word >> LINK_shift) & LINK_mask;  
  cabling.roc  = (word >> ROC_shift) & ROC_mask;
  int adc   = (word >> ADC_shift) & ADC_mask; 

  SiPixelFrameConverter::DetectorIndex detIdx = converter.toDetector(cabling);

  PixelDigi pd(detIdx.row, detIdx.col,  adc);
//  digis[detIdx.rawId].push_back(pd);

  Digis::iterator it = lower_bound(digis.begin(), digis.end(), detIdx.rawId);
  if ( (it == digis.end()) ||  (it->id != detIdx.rawId )) {               // new detunit
    it = digis.insert( it, edm::DetSet<PixelDigi>(detIdx.rawId) );
  }
  (*it).data.push_back(pd);

  theDigiCounter++;
  LogDebug("PixelDataFormatter") << print(pd);

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
  //str  <<"word64:  " << *reinterpret_cast<const bitset<64>*> (&word);
  str  <<"word64:  " << reinterpret_cast<const bitset<64>&> (word);
  return str.str();
}

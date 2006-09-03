#include "EventFilter/SiPixelRawToDigi/interface/PixelDataFormatter.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFrameConverter.h"

#include "CondFormats/SiPixelObjects/interface/PixelFEDCabling.h"
#include "CondFormats/SiPixelObjects/interface/PixelFEDLink.h"
#include "CondFormats/SiPixelObjects/interface/PixelROC.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <bitset>
#include <sstream>

using namespace std;

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
  : theNDigis(0), theCablingMap(map)
{
  int s32 = sizeof(Word32);
  int s64 = sizeof(Word64);
  int s8  = sizeof(char);
  if ( s8 != 1 || s32 != 4*s8 || s64 != 2*s32) {
     edm::LogError("**PixelDataFormatter**")
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
    int nWords = rawData.size()/8;
    LogDebug(" PixelDataFormatter ") <<"input size: "<<nWords<<" (8-bytes words)";
    if (nWords !=0) {
      const Word64 * word = reinterpret_cast<const Word64* >(rawData.data());
      for (int i=0; i<nWords; i++) {
        LogDebug("PixelDataFormatter") << "word64: "
            << *reinterpret_cast<const bitset<64>*>(word);
        static const Word64 WORD32_mask  = 0xffffffff;
        Word32 w1 =  *word >> 32 & WORD32_mask;
        Word32 w2 =  *word       & WORD32_mask;
        word2digi(converter, w1, digis);
        word2digi(converter, w2, digis);
        word++;
      }
    }
  }
  catch ( cms::Exception & err) {
    edm::LogError("PixelDataFormatter, exception") <<err.what();
  }
}


FEDRawData * PixelDataFormatter::formatData(int fedId, const Digis & digis) 
{
  vector<Word32> words;

  SiPixelFrameConverter converter(theCablingMap, fedId);
  for (Digis::const_iterator im = digis.begin(); im != digis.end(); im++) {
    uint32_t rawId = im->first;
    if ( !converter.hasDetUnit(rawId) ) continue;
    const DetDigis & detDigis = im->second;
    for (DetDigis::const_iterator it = detDigis.begin(); it != detDigis.end(); it++) {
        const PixelDigi & digi = (*it);
      try {
        digi2word( converter, rawId, digi, words);
      }
      catch ( cms::Exception& e){ edm::LogError("PixelDataFormatter, exception") <<e.what(); }
    }
  }

  //
  // since digis are writted in the form og 64-bit packets
  // add extra 32-bit word to make number of digis even
  //
  if (words.size() %2 != 0) words.push_back( Word32(0) );


  //
  // size in Bytes; create output structure
  //
  int dataSize = words.size() * sizeof(Word32);
  if (dataSize == 0) return new FEDRawData(0);
  FEDRawData * rawData = new FEDRawData(dataSize);

  //
  // write data
  //
  Word64 * word = reinterpret_cast<Word64* >(rawData->data());
  for (unsigned int i=0; i < words.size(); i+=2) {
    *word = (Word64(words[i]) << 32 ) | words[i+1];
//                <<*reinterpret_cast<bitset<32>*>(&(words[i]))
//                <<*reinterpret_cast<bitset<32>*>(&(words[i+1]))<<endl;
      LogDebug("PixelDataFormatter")  <<"word64:  "
       << *reinterpret_cast<bitset<64>*> (word);
    word++;
  }

  //
  // check memory
  //
  if (word != reinterpret_cast<Word64* >(rawData->data()+dataSize)) {
    string s = "** PROBLEM in PixelDataFormatter !!!";
    throw cms::Exception(s);
  }

  return rawData;
}

void PixelDataFormatter::interpretRawData(const PixelFEDCabling& fed, const FEDRawData& rawData, Digis& digis)
{

  try {
    int nWords = rawData.size()/8;
    LogDebug(" PixelDataFormatter ") <<"input size: "<<nWords<<" (8-bytes words)";
    if (nWords !=0) {
      const Word64 * word = reinterpret_cast<const Word64* >(rawData.data());
      for (int i=0; i<nWords; i++) {
        LogDebug("PixelDataFormatter") << "word64: " 
            << *reinterpret_cast<const bitset<64>*>(word);
        static const Word64 WORD32_mask  = 0xffffffff;
        Word32 w1 =  *word >> 32 & WORD32_mask;
        Word32 w2 =  *word       & WORD32_mask;
        word2digi(fed, w1, digis);
        word2digi(fed, w2, digis);
        word++;
      }
    }
  }
  catch ( cms::Exception & err) {
    edm::LogError("PixelDataFormatter, exception") <<err.what();
  }
}

 
FEDRawData * PixelDataFormatter::formatData( 
    const PixelFEDCabling & fed, 
    const Digis & digis)
{

  vector<Word32> words;
  for (int idxLink = 0; idxLink < fed.numberOfLinks(); idxLink++) {
    const PixelFEDLink * link = fed.link(idxLink);
    int linkid = link->id();
    int numberOfRocs = link->numberOfROCs();
    for(int idxRoc = 0; idxRoc < numberOfRocs; idxRoc++) {
      const PixelROC * roc = link->roc(idxRoc);
      Digis::const_iterator im= digis.find(roc->rawId());
      if (im == digis.end() ) continue;
      Range range(im->second.begin(), im->second.end());
      for (DetDigis::const_iterator it = range.first; it != range.second;it++) {
        const PixelDigi & digi = (*it);
        digi2word(linkid, *roc, digi, words);
      }
    }
  }

  //
  // debug only
  //
//{
//  if (numDigi != words.size() ) {
//    cout << " ** HERE PixelDataFormatter** PROBLEM !!!!"
//               <<" numDigi: "<< numDigi
//               <<" words.size(): " << words.size()
//               <<", send exception" << endl;
//    throw Genexception("** PixelDataFormatter, numDigi != words.size()");
//  }
//}
 
  //
  // since digis are writted in the form og 64-bit packets
  // add extra 32-bit word to make number of digis even
  //
  if (words.size() %2 != 0) words.push_back( Word32(0) );


  //
  // size in Bytes; create output structure
  //
  int dataSize = words.size() * sizeof(Word32);
  if (dataSize == 0) return new FEDRawData(0);
  FEDRawData * rawData = new FEDRawData(dataSize);

  //
  // write data
  //
  Word64 * word = reinterpret_cast<Word64* >(rawData->data());
  for (unsigned int i=0; i < words.size(); i+=2) {
    *word = (Word64(words[i]) << 32 ) | words[i+1];
//                <<*reinterpret_cast<bitset<32>*>(&(words[i]))
//                <<*reinterpret_cast<bitset<32>*>(&(words[i+1]))<<endl;
      LogDebug("PixelDataFormatter")  <<"word64:  "
       << *reinterpret_cast<bitset<64>*> (word);
    word++;
  }

  //
  // check memory
  //
  if (word != reinterpret_cast<Word64* >(rawData->data()+dataSize)) {
    string s = "** PROBLEM in PixelDataFormatter !!!";
    throw cms::Exception(s);
  }

  return rawData;
}
 


void PixelDataFormatter::digi2word(
    int linkId,
    const PixelROC &roc, 
    const PixelDigi & pd,
    vector<Word32> &words) const
{
    PixelROC::GlobalPixel glo = { pd.row(), pd.column() };
    PixelROC::LocalPixel  loc = roc.toLocal(glo);
    if (! roc.inside(loc) ) return;
    LogDebug("PixelDataFormatter")<< "DIGI: row: " << pd.row()
               <<", col: " << pd.column()
               <<", adc: " << pd.adc() ;

    theNDigis++;

    Word32 word = 
             (linkId << LINK_shift)
           | (roc.idInLink() << ROC_shift)
           | (loc.dcol << DCOL_shift)
           | (loc.pxid << PXID_shift)
           | (pd.adc() << ADC_shift);
    words.push_back(word);
}

void PixelDataFormatter::digi2word( const SiPixelFrameConverter& converter,
    uint32_t detId, const PixelDigi& digi, std::vector<Word32> & words) const
{
  LogDebug("PixelDataFormatter")<< "DIGI: row: " << digi.row()
               <<", col: " << digi.column()
               <<", adc: " << digi.adc() ;

  SiPixelFrameConverter::DetectorIndex detector = {detId, digi.row(), digi.column()};
  SiPixelFrameConverter::CablingIndex  cabling = converter.toCabling(detector);
  theNDigis++;

  Word32 word =
             (cabling.link  << LINK_shift)
           | (cabling.roc << ROC_shift)
           | (cabling.dcol << DCOL_shift)
           | (cabling.pxid << PXID_shift)
           | (digi.adc() << ADC_shift);
    words.push_back(word);
}

void PixelDataFormatter::word2digi(const SiPixelFrameConverter& converter, 
    const Word32 & word, Digis & digis) const
{
  // do not interpret false digis
  if (word == 0 ) return;

  theNDigis++;
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
  digis[detIdx.rawId].push_back(pd);

  LogDebug("PixelDataFormatter")<< "DIGI: row: " << pd.row()
                          <<", col: " << pd.column()
                          <<", adc: " << pd.adc();
}


void PixelDataFormatter::word2digi(const PixelFEDCabling & fed, 
    const Word32 & word, Digis & digis) const
{
  // do not interpret false digis
  if (word == 0 ) return;

  theNDigis++;
  static const Word32 LINK_mask = ~(~Word32(0) << LINK_bits);
  static const Word32 ROC_mask  = ~(~Word32(0) << ROC_bits);
  static const Word32 DCOL_mask = ~(~Word32(0) << DCOL_bits);
  static const Word32 PXID_mask = ~(~Word32(0) << PXID_bits);
  static const Word32 ADC_mask  = ~(~Word32(0) << ADC_bits);

  PixelROC::LocalPixel loc;
  loc.dcol = (word >> DCOL_shift) & DCOL_mask;
  loc.pxid = (word >> PXID_shift) & PXID_mask;
  int ilink = (word >> LINK_shift) & LINK_mask;
  int iroc   = (word >> ROC_shift) & ROC_mask;
  int adc   = (word >> ADC_shift) & ADC_mask;


  const PixelFEDLink * link = fed.link(ilink);
  if (!link) {
    stringstream stm;
    stm << "FED shows no link of id= " << ilink;
    throw cms::Exception(stm.str());
  }
  const PixelROC * roc = link->roc(iroc);
  if (!roc) {
    stringstream stm;
    stm << "Link=" << ilink << " shows no ROC with id=" << iroc;
    throw cms::Exception(stm.str());
  }

  PixelROC::GlobalPixel glo = roc->toGlobal(loc);
  PixelDigi pd( glo.row, glo.col, adc);


  LogDebug("PixelDataFormatter")<< "DIGI: row: " << pd.row()
                          <<", col: " << pd.column()
                          <<", adc: " << pd.adc();
  //
  //
  uint32_t detid = roc->rawId();
  digis[detid].push_back(pd);
}


#include "EventFilter/RPCRawToDigi/interface/RPCRawDataPacker.h"

#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"

#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/RPCDigi/interface/RPCDigi.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "EventFilter/RPCRawToDigi/interface/RPCRecordFormatter.h"

#include "CondFormats/RPCObjects/interface/RPCReadOutMapping.h"
#include "CondFormats/RPCObjects/interface/ChamberRawDataSpec.h"
#include "CondFormats/RPCObjects/interface/LinkBoardChannelCoding.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <bitset>

using namespace std;
using namespace edm;

FEDRawData * RPCRawDataPacker::rawData( int fedId, const RPCDigiCollection * digis, const RPCRecordFormatter & formatter)
{
  typedef  DigiContainerIterator<RPCDetId, RPCDigi> DigiRangeIterator;
  vector<Word64> dataWords;
  RPCRecordFormatter::Record bxRecord, tbRecord, lbRecord;
  RPCRecordFormatter::Record empty;  formatter.setEmptyRecord(empty);

  int trigger_BX = 200;   // FIXME - set event by event but correct bx assigment in digi 
  for (DigiRangeIterator it=digis->begin(); it != digis->end(); it++) {
    RPCDetId rpcDetId = (*it).first;
    uint32_t rawDetId = rpcDetId.rawId();
    RPCDigiCollection::Range range = digis->get(rpcDetId);
    for (vector<RPCDigi>::const_iterator  id = range.first; id != range.second; id++) {
      const RPCDigi & digi = (*id);
      LogInfo("RPCRawDataPacker")<<"detId: "<<rawDetId<<" digi: "<< digi;
      int statusOK = formatter.pack(rawDetId, digi, trigger_BX, bxRecord, tbRecord, lbRecord);
      Word64 w = (((Word64(bxRecord) << 16) | tbRecord) << 16 | lbRecord ) << 16 |empty ;
      if (statusOK) dataWords.push_back(w); 
    }
  }
  // merge data words

  //
  // create raw data
  //
  int nHeaders = 1;
  int nTrailers = 1;
  int dataSize = (nHeaders+nTrailers+dataWords.size()) * sizeof(Word64);
  FEDRawData * raw = new FEDRawData(dataSize);

  //
  // add header
  //
  unsigned char *pHeader  = raw->data(); 
  int evt_ty = 3;
  int lvl1_ID = 100; // FIXME - increase/set event by event 
  int source_ID = fedId;
  FEDHeader::set(pHeader, evt_ty, lvl1_ID, trigger_BX, source_ID); 

  //
  // add datawords
  //
  for (unsigned int idata = 0; idata < dataWords.size(); idata ++) {
    Word64 * word = reinterpret_cast<Word64* >(pHeader+(idata+1)*sizeof(Word64));
    *word = dataWords[idata];
  }
  
  //
  // add trailer
  //
  unsigned char *pTrailer = pHeader + raw->size()-sizeof(Word64);
  int crc = 0;
  int evt_stat = 15;
  int tts = 15;
  int datasize =  raw->size()/sizeof(Word64);
  FEDTrailer::set(pTrailer, datasize, crc, evt_stat, tts);
  
  return raw;
}


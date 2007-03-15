#include "EventFilter/RPCRawToDigi/interface/RPCRawDataPacker.h"

#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"

#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/RPCDigi/interface/RPCDigi.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"

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

////////////////////////////
#include "EventFilter/RPCRawToDigi/interface/RPCRawDataPattern.h"
////////////////////////////
using namespace std;
using namespace edm;

bool RPCRawDataPacker::Records::samePartition(const Records & r) const
{
  if (this->bx != r.bx) return false;
  if (this->tb != r.tb) return false;
  RPCRecordFormatter::Record mask = 0xFF << 8;
  RPCRecordFormatter::Record lb1 = this->lb & mask;
  RPCRecordFormatter::Record lb2 = r.lb & mask;
  if (lb1 != lb2) return false;
//  LogTrace("")<<"LBRECORD 1:  " <<*reinterpret_cast<const bitset<16>*>(& (this->lb));
//  LogTrace("")<<"LBRECORD 2:  " <<*reinterpret_cast<const bitset<16>*>(& (r.lb));
  return true;
}

RPCRawDataPacker::RPCRawDataPacker(){

  //set vector sizes
  int LAST_BX = 15;
  int FIRST_BX = 0;
  vector<RPCPacData> rpdv(18,RPCPacData(0));
  vector<vector<RPCPacData> > rpdvv(LAST_BX-FIRST_BX,rpdv);
  linkData_ = rpdvv;

  myXMLWriter = new  LinkDataXMLWriter();

}

RPCRawDataPacker::~RPCRawDataPacker(){ delete myXMLWriter; }

LinkDataXMLWriter * RPCRawDataPacker::getXMLWriter() const {return myXMLWriter;}

std::vector<RPCRawDataPacker::Records> RPCRawDataPacker::margeRecords(const std::vector<Records> & data) const
{
  std::vector<Records> result;
  typedef vector<Records>::const_iterator ICR;
  typedef vector<Records>::iterator IR;
  for (ICR id=data.begin(), idEnd = data.end(); id != idEnd; ++id) {
    bool merged = false;
    for (IR ir = result.begin(), irEnd = result.end(); ir != irEnd; ++ir) {
      Records & records = *ir;
      if (id->samePartition( records)) {
        LogTrace("")<<" merging...."<<endl;
        records.lb |= id->lb;
//  LogTrace("")<<"LBRECORD MRG:" <<*reinterpret_cast<const bitset<16>*>(& (records.lb));
        merged = true;
      }
    } 
    if (!merged) result.push_back(*id);
  } 
  return result;
}

FEDRawData * RPCRawDataPacker::rawData( int fedId, const RPCDigiCollection * digis, const RPCRecordFormatter & formatter)
{
  typedef  DigiContainerIterator<RPCDetId, RPCDigi> DigiRangeIterator;
  vector<Records> dataRecords;
  RPCRecordFormatter::Record empty;  formatter.setEmptyRecord(empty);
  Records records;
  RPCRecordFormatter::Record & bxRecord = records.bx;
  RPCRecordFormatter::Record & tbRecord = records.tb;
  RPCRecordFormatter::Record & lbRecord = records.lb;

  LogDebug("RPCRawDataPacker")<<"Packing Fed id="<<fedId;
  int trigger_BX = 200;   // FIXME - set event by event but correct bx assigment in digi 
  for (DigiRangeIterator it=digis->begin(); it != digis->end(); it++) {
    RPCDetId rpcDetId = (*it).first;
    uint32_t rawDetId = rpcDetId.rawId();
    RPCDigiCollection::Range range = digis->get(rpcDetId);
    for (vector<RPCDigi>::const_iterator  id = range.first; id != range.second; id++) {
      const RPCDigi & digi = (*id);
//      LogDebug("RPCRawDataPacker")<<"detId: "<<rawDetId<<" digi: "<< digi;
      int statusOK = formatter.pack(rawDetId, digi, trigger_BX, bxRecord, tbRecord, lbRecord);
      if (statusOK) dataRecords.push_back(records);
    }
  }
  //
  // merge data words
  //
  LogTrace("RPCRawDataPacker") <<" size of   data: " << dataRecords.size();
  vector<Records> merged = margeRecords(dataRecords); 
  LogTrace("") <<" size of megred: " << merged.size();

  //
  // create data words
  //
  
  vector<Word64> dataWords;
  typedef vector<Records>::const_iterator IR;
  for (IR ir = merged.begin(), irEnd =  merged.end() ; ir != irEnd; ++ir) {
    Word64 w = (((Word64(ir->bx) << 16) | ir->tb) << 16 | ir->lb) << 16 |empty;
    dataWords.push_back(w); 
    //////////////////////////////////////////////
    int triggerCrateNum = 0;
    int triggerBoardNum = 0;
    int opticalLinkNum = ( ir->tb >> rpcraw::tb_link::TB_LINK_INPUT_NUMBER_SHIFT )& rpcraw::tb_link::TB_LINK_INPUT_NUMBER_MASK;    
    int partitionData= (ir->lb >>rpcraw::lb::PARTITION_DATA_SHIFT)&rpcraw::lb::PARTITION_DATA_MASK;
    int halfP = (ir->lb >> rpcraw::lb::HALFP_SHIFT ) & rpcraw::lb::HALFP_MASK;
    int eod = (ir->lb >> rpcraw::lb::EOD_SHIFT ) & rpcraw::lb::EOD_MASK;
    int partitionNumber = (ir->lb >> rpcraw::lb::PARTITION_NUMBER_SHIFT ) & rpcraw::lb::PARTITION_NUMBER_MASK;
    int lbNumber = (ir->lb >> rpcraw::lb::LB_SHIFT ) & rpcraw::lb::LB_MASK ;
    myXMLWriter->addLinkData(triggerCrateNum, triggerBoardNum , opticalLinkNum, 
			     lbNumber, partitionNumber,  partitionData, halfP, eod);
    //////////////////////////////////////////////
    
  }

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
  int tts = 0;
  int datasize =  raw->size()/sizeof(Word64);
  FEDTrailer::set(pTrailer, datasize, crc, evt_stat, tts);
  
  return raw;
}


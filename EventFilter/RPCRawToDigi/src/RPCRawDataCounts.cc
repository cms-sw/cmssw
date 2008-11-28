#include "EventFilter/RPCRawToDigi/interface/RPCRawDataCounts.h"
#include "EventFilter/RPCRawToDigi/interface/DataRecord.h"
#include "EventFilter/RPCRawToDigi/interface/RecordSLD.h"
#include "EventFilter/RPCRawToDigi/interface/ErrorRDDM.h"
#include "EventFilter/RPCRawToDigi/interface/ErrorRDM.h"
#include "EventFilter/RPCRawToDigi/interface/ErrorRCDM.h"
#include "EventFilter/RPCRawToDigi/interface/ReadoutError.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <vector>
#include <iostream>
#include <sstream>

#include "TH1F.h"
#include "TH2F.h"

using namespace rpcrawtodigi;
using namespace std;

typedef std::map< std::pair<int,int>, int >::const_iterator IT;

void RPCRawDataCounts::addDccRecord(int fed, const rpcrawtodigi::DataRecord & record, int weight)
{
  DataRecord::DataRecordType type = record.type();
  switch (type) {
    case (DataRecord::StartOfTbLinkInputNumberData) : { theGoodEvents[make_pair(fed, RecordSLD(record).rmb())] += weight; break; }
    case (DataRecord::RDDM)                         : { theBadEvents[make_pair(fed,ErrorRDDM(record).rmb())]   += weight; break;}
    case (DataRecord::RDM)                          : { theBadEvents[make_pair(fed,ErrorRDM(record).rmb())]    += weight; break;}
    case (DataRecord::RCDM)                         : { theBadEvents[make_pair(fed,ErrorRCDM(record).rmb())]   += weight; break;}
    default : {}
  }
  
  theRecordTypes[ make_pair(fed,type) ] += weight;
}

void RPCRawDataCounts::addReadoutError(int fed, const rpcrawtodigi::ReadoutError & e, int weight)
{
  theReadoutErrors[ make_pair(fed,e.type()) ] +=  weight;
}

void RPCRawDataCounts::operator+= (const RPCRawDataCounts & o)
{

  for (IT irt= o.theRecordTypes.begin(); irt != o.theRecordTypes.end(); ++irt) {
    theRecordTypes[ make_pair(irt->first.first,irt->first.second) ] += irt->second;
  }

  for (IT ire=o.theReadoutErrors.begin(); ire != o.theReadoutErrors.end();++ire) {
    theReadoutErrors[ make_pair(ire->first.first,ire->first.second) ] += ire->second;
  }

  for (IT ire=o.theGoodEvents.begin(); ire != o.theGoodEvents.end();++ire) {
    theGoodEvents[ make_pair(ire->first.first,ire->first.second) ] += ire->second;
  }

  for (IT ire=o.theBadEvents.begin(); ire != o.theBadEvents.end();++ire) {
    theBadEvents[ make_pair(ire->first.first,ire->first.second) ] += ire->second;
  }
}

std::string RPCRawDataCounts::print() const 
{
  std::ostringstream str;
  for (IT irt=theRecordTypes.begin(); irt != theRecordTypes.end(); ++irt) {
    str << "RECORD ("<<irt->first.first<<","<<irt->first.second<<")"<<irt->second;
  }
  for (IT ire=theReadoutErrors.begin(); ire != theReadoutErrors.end();++ire) {
    str <<"ERROR("<<ire->first.first<<","<<ire->first.second<<")="<<ire->second<<endl;
  } 
  return str.str();
}

TH1F * RPCRawDataCounts::emptyRecordTypeHisto(int fedId) const {
  std::ostringstream str;
  str <<"recordType_"<<fedId;
  TH1F * result = new TH1F(str.str().c_str(),str.str().c_str(),9, 0.5,9.5);
  result->SetTitleOffset(1.4,"x"); 
  for (unsigned int i=1; i<=9; ++i) {
    DataRecord::DataRecordType code = static_cast<DataRecord::DataRecordType>(i);
    result->GetXaxis()->SetBinLabel(i,DataRecord::name(code).c_str());
  }
  return result;
}

TH1F * RPCRawDataCounts::emptyReadoutErrorHisto(int fedId) const {
  std::ostringstream str;
  str <<"readoutErrors_"<<fedId;
  TH1F * result = new TH1F(str.str().c_str(),str.str().c_str(),8, 0.5,8.5);
  for (unsigned int i=1; i<=8; ++i) {
    ReadoutError::ReadoutErrorType code =  static_cast<ReadoutError::ReadoutErrorType>(i);
    result->GetXaxis()->SetBinLabel(i,ReadoutError::name(code).c_str());
  }
  return result;
}

void RPCRawDataCounts::fillRecordTypeHisto(int fedId, TH1F* histo) const
{
  for (IT irt=theRecordTypes.begin(); irt != theRecordTypes.end(); ++irt) {
    if (irt->first.first != fedId) continue;
    histo->Fill(irt->first.second,irt->second);
  }
}

void RPCRawDataCounts::fillReadoutErrorHisto(int fedId, TH1F* histo) const
{
  for (IT ire=theReadoutErrors.begin(); ire != theReadoutErrors.end(); ++ire) {
    if (ire->first.first != fedId) continue;
    histo->Fill(ire->first.second,ire->second);
  }
}

void RPCRawDataCounts::fillGoodEventsHisto(TH2F* histo) const
{
   for (IT it = theGoodEvents.begin(); it != theGoodEvents.end(); ++it)
       histo->Fill(it->first.second, it->first.first, it->second);
}

void RPCRawDataCounts::fillBadEventsHisto(TH2F* histo) const
{
   for (IT it = theBadEvents.begin(); it != theBadEvents.end(); ++it)
       histo->Fill(it->first.second, it->first.first, it->second);
}


#include "DQM/RPCMonitorClient/interface/RPCRawDataCountsHistoMaker.h"
#include "EventFilter/RPCRawToDigi/interface/RPCRawDataCounts.h"
#include "EventFilter/RPCRawToDigi/interface/DataRecord.h"
#include "EventFilter/RPCRawToDigi/interface/ReadoutError.h"

#include "TH1F.h"
#include "TH2F.h"

#include <vector>
#include <sstream>
using namespace rpcrawtodigi;
using namespace std;


typedef std::map< std::pair<int,int>, int >::const_iterator IT;

TH1F * RPCRawDataCountsHistoMaker::emptyReadoutErrorHisto(int fedId) {
  ostringstream str;
  str <<"readoutErrors_"<<fedId;
  TH1F * result = new TH1F(str.str().c_str(),str.str().c_str(),9, 0.5,9.5);
  for (unsigned int i=1; i<=9; ++i) {
    ReadoutError::ReadoutErrorType code =  static_cast<ReadoutError::ReadoutErrorType>(i);
    result->GetXaxis()->SetBinLabel(i,ReadoutError::name(code).c_str());
  }
  return result;
}

TH1F * RPCRawDataCountsHistoMaker::emptyRecordTypeHisto(int fedId) {
  ostringstream str;
  str <<"recordType_"<<fedId;
  TH1F * result = new TH1F(str.str().c_str(),str.str().c_str(),9, 0.5,9.5);
  result->SetTitleOffset(1.4,"x");
  for (unsigned int i=1; i<=9; ++i) {
    DataRecord::DataRecordType code = static_cast<DataRecord::DataRecordType>(i);
    result->GetXaxis()->SetBinLabel(i,DataRecord::name(code).c_str());
  }
  return result;
}


 map< pair<int,int>, int > RPCRawDataCountsHistoMaker::readoutErrors(void){

   return theCounts.theReadoutErrors;   
}


 map< pair<int,int>, int > RPCRawDataCountsHistoMaker::recordTypes(void){

   return theCounts.theRecordTypes;   
}




void RPCRawDataCountsHistoMaker::fillRecordTypeHisto(int fedId, TH1F* histo) const
{
  for (IT irt=theCounts.theRecordTypes.begin(); irt != theCounts.theRecordTypes.end(); ++irt) {
    if (irt->first.first != fedId) continue;
    histo->Fill(irt->first.second,irt->second);
  }
}

void RPCRawDataCountsHistoMaker::fillReadoutErrorHisto(int fedId, TH1F* histo) const
{
  for (IT ire=theCounts.theReadoutErrors.begin(); ire != theCounts.theReadoutErrors.end(); ++ire) {
    if (ire->first.first != fedId) continue;
    histo->Fill(ire->first.second,ire->second);
  }
}

void RPCRawDataCountsHistoMaker::fillGoodEventsHisto(TH2F* histo) const
{
   for (IT it = theCounts.theGoodEvents.begin(); it != theCounts.theGoodEvents.end(); ++it)
       histo->Fill(it->first.second, it->first.first, it->second);
}

void RPCRawDataCountsHistoMaker::fillBadEventsHisto(TH2F* histo) const
{
   for (IT it = theCounts.theBadEvents.begin(); it != theCounts.theBadEvents.end(); ++it)
       histo->Fill(it->first.second, it->first.first, it->second);
}



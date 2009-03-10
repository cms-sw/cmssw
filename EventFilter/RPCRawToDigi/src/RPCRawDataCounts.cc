#include "EventFilter/RPCRawToDigi/interface/RPCRawDataCounts.h"
#include "EventFilter/RPCRawToDigi/interface/DataRecord.h"

#include <vector>
#include <iostream>
#include <sstream>

using namespace rpcrawtodigi;
using namespace std;

  typedef std::map<int,int>::const_iterator IRE;
  typedef std::map<int, std::vector<int> >::const_iterator IRT;

void RPCRawDataCounts::addRecordType(int fed, int type, int weight)
{
  if (theRecordTypes.find(fed) == theRecordTypes.end()) {
    theRecordTypes[fed]=vector<int>( 10,0);
  }
  vector<int> & v = theRecordTypes[fed]; 
  v[type] += weight;
}

void RPCRawDataCounts::addReadoutError(int error, int weight)
{
  if ( theReadoutErrors.find(error) == theReadoutErrors.end() ) theReadoutErrors[error]=0;
  theReadoutErrors[error] += weight;
}

void RPCRawDataCounts::operator+= (const RPCRawDataCounts & o)
{
  for (IRE ire=o.theReadoutErrors.begin(); ire != o.theReadoutErrors.end();++ire)
  {
    addReadoutError(ire->first,ire->second);
  }
  for (IRT irt= o.theRecordTypes.begin(); irt != o.theRecordTypes.end(); ++irt) {
    int fed = irt->first;
    const vector<int> & v = irt->second;
    for (unsigned int itype=0; itype < v.size(); itype++) 
      addRecordType(fed, static_cast<int>(itype), v[itype]);
  }
}

std::string RPCRawDataCounts::print() const 
{
  std::ostringstream str;
  for (IRT irt=theRecordTypes.begin(); irt != theRecordTypes.end(); ++irt) {
    str << "FED: "<<irt->first<<" ";
    const vector<int> & v = irt->second;
    for (unsigned int itype=0; itype < v.size(); itype++) str <<v[itype]<<", ";
    str << endl; 
  }
  for (IRE ire=theReadoutErrors.begin(); ire != theReadoutErrors.end();++ire) {
    str <<"ERROR("<<ire->first<<")="<<ire->second<<endl;
  } 
  return str.str();
}


void RPCRawDataCounts::recordTypeVector(int fedId, std::vector<double>& out) const {
  out.clear();
  IRT irt = theRecordTypes.find(fedId);
  if (irt != theRecordTypes.end()) {
    const vector<int> & v = irt->second;
    for (int i=1; i<=9; ++i) {
      out.push_back(double(i));
      out.push_back(v[i]);
    }
  }
}

void RPCRawDataCounts::readoutErrorVector(std::vector<double>& out) const {
  out.clear();
  for (int i=1; i<9; ++i) {
    IRE ire = theReadoutErrors.find(i);
    if(ire != theReadoutErrors.end()) {
     out.push_back(ire->first);
     out.push_back(ire->second);
    }
  }
}


TH1F * RPCRawDataCounts::recordTypeHisto(int fedId) const {
  std::ostringstream str;
  str <<"recordType_"<<fedId;
  TH1F * result = new TH1F(str.str().c_str(),str.str().c_str(),9, 0.5,9.5);
  result->SetTitleOffset(1.4,"x"); 
  for (unsigned int i=1; i<=9; ++i) {
    DataRecord::recordName code = static_cast<DataRecord::recordName>(i);
    result->GetXaxis()->SetBinLabel(i,DataRecord::name(code).c_str());
  }

  IRT irt = theRecordTypes.find(fedId);
  if (irt != theRecordTypes.end()) {
    const vector<int> & v = irt->second;
    for (int i=1; i<=9; ++i) result->Fill(float(i),v[i]);
  } 
  return result;
}

TH1F * RPCRawDataCounts::readoutErrorHisto() const {
  std::ostringstream str;
  str <<"readoutErrors";
  TH1F * result = new TH1F(str.str().c_str(),str.str().c_str(),8, 0.5,8.5);
  for (unsigned int i=1; i<=8; ++i) {
    RPCRawDataCounts::ReadoutError code =  static_cast<RPCRawDataCounts::ReadoutError>(i);
    result->GetXaxis()->SetBinLabel(i,readoutErrorName(code).c_str());
  }
  for (int i=1; i<9; ++i) {
    IRE ire = theReadoutErrors.find(i);
    if(ire != theReadoutErrors.end()) result->Fill(ire->first,ire->second);  
  }
  return result;
}

std::string RPCRawDataCounts:: readoutErrorName(const ReadoutError & code)
{
  std::string result;
  switch (code) {
    case (HeaderCheckFail)      : { result = "HeaderCheckFail"; break; }
    case (InconsitentFedId)     : { result = "InconsitentFedId"; break; }
    case (TrailerCheckFail)     : { result = "TrailerCheckFail"; break; }
    case (InconsistentDataSize) : { result = "InconsistentDataSize"; break; }
    case (InvalidLB)            : { result = "InvalidLB"; break; }
    case (EmptyPackedStrips)    : { result = "EmptyPackedStrips"; break; }
    case (InvalidDetId)         : { result = "InvalidDetId"; break; }
    case (InvalidStrip)         : { result = "InvalidStrip"; break; }
    default                     : { result = "NoProblem"; } 
  }
  return result;
}

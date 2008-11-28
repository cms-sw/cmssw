#include "EventFilter/RPCRawToDigi/interface/DataRecord.h"
#include "EventFilter/RPCRawToDigi/interface/RecordBX.h"
#include "EventFilter/RPCRawToDigi/interface/RecordSLD.h"
#include "EventFilter/RPCRawToDigi/interface/RecordCD.h"
#include "EventFilter/RPCRawToDigi/interface/EmptyWord.h"
#include "EventFilter/RPCRawToDigi/interface/ErrorRDM.h"
#include "EventFilter/RPCRawToDigi/interface/ErrorSDDM.h"
#include "EventFilter/RPCRawToDigi/interface/ErrorRDDM.h"
#include "EventFilter/RPCRawToDigi/interface/ErrorRCDM.h"

using namespace rpcrawtodigi;

DataRecord::DataRecordType  DataRecord::type() const 
{
  DataRecordType wordType = UndefinedType;
  if (RecordBX::matchType(*this)) wordType = StartOfBXData;
  if (RecordSLD::matchType(*this)) wordType = StartOfTbLinkInputNumberData;
  if (RecordCD::matchType(*this)) wordType = ChamberData;
  if (EmptyWord::matchType(*this)) wordType = Empty;
  if (ErrorRCDM::matchType(*this)) wordType = RCDM;
  if (ErrorSDDM::matchType(*this)) wordType = SDDM;
  if (ErrorRDDM::matchType(*this)) wordType = RDDM;
  if (ErrorRDM::matchType(*this))  wordType = RDM;

  return wordType;
}

std::string DataRecord::print(const DataRecord & record) 
{
  std::ostringstream str;
  
  if (RecordBX::matchType(record)) return RecordBX(record).print();
  if (RecordSLD::matchType(record)) return RecordSLD(record).print();
  if (RecordCD::matchType(record)) return RecordCD(record).print();
  if (EmptyWord::matchType(record)) return EmptyWord().print();
  if (ErrorRCDM::matchType(record)) return ErrorRCDM(record).print();
  if (ErrorSDDM::matchType(record)) return ErrorSDDM(record).print();
  if (ErrorRDDM::matchType(record)) return ErrorRDDM(record).print();
  if (ErrorRDM::matchType(record))  return ErrorRDM(record).print();

  return str.str();
}

std::string DataRecord::name(const DataRecordType& code)
{
  std::string result;
  switch (code) {
    case (None)                         : { result = "None"; break; }
    case (StartOfBXData)                : { result = "StartOfBXData"; break; }
    case (StartOfTbLinkInputNumberData) : { result = "StartOfTBLnkData"; break; }
    case (ChamberData)                  : { result = "ChamberData"; break; }
    case (Empty)                        : { result = "Empty"; break; }
    case (RDDM)                         : { result = "RDDM"; break; }
    case (SDDM)                         : { result = "SDDM"; break; }
    case (RCDM)                         : { result = "RCDM"; break; }
    case (RDM)                          : { result = "RDM"; break; }
    default                             : { result = "UndefinedType"; } 
  }
  return result;
}

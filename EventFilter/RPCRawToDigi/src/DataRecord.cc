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

DataRecord::recordName  DataRecord::type() const 
{
  recordName wordType = UndefinedType;
  if (RecordBX::matchType(theData)) wordType = StartOfBXData;
  if (RecordSLD::matchType(theData)) wordType = StartOfTbLinkInputNumberData;
  if (RecordCD::matchType(theData)) wordType = ChamberData;
  if (EmptyWord::matchType(theData)) wordType = Empty;
  if (ErrorRCDM::matchType(theData)) wordType = RCDM;
  if (ErrorSDDM::matchType(theData)) wordType = SDDM;
  if (ErrorRDDM::matchType(theData)) wordType = RDDM;
  if (ErrorRDM::matchType(theData))  wordType = RDM;
  return wordType;
}

std::string DataRecord::print(const DataRecord & data) 
{
  std::ostringstream str;
  
  if (RecordBX::matchType(data)) return RecordBX(data).print();
  if (RecordSLD::matchType(data)) return RecordSLD(data).print();
  if (RecordCD::matchType(data)) return RecordCD(data).print();
  if (EmptyWord::matchType(data)) return EmptyWord().print();
  if (ErrorRCDM::matchType(data)) return ErrorRCDM(data).print();
  if (ErrorSDDM::matchType(data)) return ErrorSDDM().print();
  if (ErrorRDDM::matchType(data)) return ErrorRDDM(data).print();
  if (ErrorRDM::matchType(data))  return ErrorRDM(data).print();

  return str.str();
}

std::string DataRecord::name(const recordName & code)
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

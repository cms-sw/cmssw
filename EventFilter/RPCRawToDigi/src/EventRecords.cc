#include "EventFilter/RPCRawToDigi/interface/EventRecords.h"
#include "DataFormats/RPCDigi/interface/ErrorRCDM.h"
#include "DataFormats/RPCDigi/interface/ErrorRDDM.h"
#include "DataFormats/RPCDigi/interface/ErrorRDM.h"
#include "DataFormats/RPCDigi/interface/ErrorSDDM.h"


using namespace rpcrawtodigi;
using std::vector;

int EventRecords::dataToTriggerDelay() const
{
  static const int nOrbits = 3564;
  if (!complete()) return nOrbits;
  int diff = recordBX().bx() - triggerBx() + 3;
  if (diff >  nOrbits/2) diff -= nOrbits;
  if (diff < -nOrbits/2) diff += nOrbits;
  return diff;
}


void EventRecords::add(const DataRecord & record)
{
  
  if (record.type() == DataRecord::StartOfBXData) {
    theRecordBX = RecordBX(record);
    theValidBX = true;
    theValidLN = false;
    theValidCD = false;
    theErrors.clear();
  }
  else if (record.type() == DataRecord::StartOfTbLinkInputNumberData) {
    theRecordSLD = RecordSLD(record);
    theValidLN = true;
    theValidCD = false;
  }
  else if (record.type() == DataRecord::ChamberData) {
    theRecordCD = RecordCD(record);
    theValidCD = true;
  } 
  else {
//    theValidBX = false;
//    theValidLN = false;
    theValidCD = false;
    if ( record.type() > DataRecord::Empty) theErrors.push_back(record);
  }
}

bool EventRecords::samePartition(const EventRecords & r) const
{
  if (this->recordBX().data() != r.recordBX().data() ) return false;
  if (this->recordSLD().data() != r.recordSLD().data() ) return false;
  typedef DataRecord::Data Record; 
  Record mask = 0xFF << 8;
  Record lb1 = this->recordCD().data() & mask;
  Record lb2 = r.recordCD().data() & mask;
  if (lb1 != lb2) return false;
  return true;
}

vector<EventRecords> EventRecords::mergeRecords(const vector<EventRecords> & data)
{
  std::vector<EventRecords> result;
  typedef vector<EventRecords>::const_iterator ICR;
  typedef vector<EventRecords>::iterator IR;
  for (ICR id= data.begin(), idEnd = data.end(); id != idEnd; ++id) {
    bool merged = false;
    for (IR ir = result.begin(), irEnd = result.end(); ir != irEnd; ++ir) {
      EventRecords & event = *ir;
      if (id->samePartition( event)) {
        DataRecord::Data lbd = event.recordCD().data();
        lbd |= id->recordCD().data();
        event.add( RecordCD(lbd) );
        merged = true;
      }
    }
    if (!merged) result.push_back(*id);
  }
  return result;

}

std::string EventRecords::print(const DataRecord::DataRecordType& type) const
{
  std::ostringstream str;
  str <<" ==>";
  if (type == DataRecord::StartOfBXData && theValidBX)               str << theRecordBX.print(); 
  if (type == DataRecord::StartOfTbLinkInputNumberData&& theValidLN) str << theRecordSLD.print(); 
  if (type == DataRecord::ChamberData && theValidCD)               str << theRecordCD.print();
  if (type == DataRecord::Empty)                                   str <<" EPMTY";
  for (vector<DataRecord>::const_iterator ie=theErrors.begin(); ie < theErrors.end(); ++ie) { 
    if (type == DataRecord::RDDM)   str << ErrorRDDM(*ie).print(); 
    if (type == DataRecord::SDDM)   str << ErrorSDDM(*ie).print(); 
    if (type == DataRecord::RCDM)   str << ErrorRCDM(*ie).print(); 
    if (type == DataRecord::RDM)   str << ErrorRDM(*ie).print(); 
  }
  return str.str();
}

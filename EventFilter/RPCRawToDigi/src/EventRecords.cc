#include "EventFilter/RPCRawToDigi/interface/EventRecords.h"


using namespace rpcrawtodigi;
using std::vector;


void EventRecords::add(const DataRecord & record)
{
  
  if (record.type() == DataRecord::StartOfBXData) {
    theBXRecord = BXRecord(record);
    theValidBX = true;
    theValidTB = false;
    theValidLB = false;
  }
  else if (record.type() == DataRecord::StartOfTbLinkInputNumberData) {
    theTBRecord = TBRecord(record);
    theValidTB = true;
    theValidLB = false;
  }
  else if (record.type() == DataRecord::LinkBoardData) {
    theLBRecord = LBRecord(record);
    theValidLB = true;
  } 
  else {
//    theValidBX = false;
//    theValidTB = false;
    theValidLB = false;
  }
}

bool EventRecords::samePartition(const EventRecords & r) const
{
  if (this->bxRecord().data() != r.bxRecord().data() ) return false;
  if (this->tbRecord().data() != r.tbRecord().data() ) return false;
  typedef DataRecord::RecordType Record; 
  Record mask = 0xFF << 8;
  Record lb1 = this->lbRecord().data() & mask;
  Record lb2 = r.lbRecord().data() & mask;
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
        DataRecord::RecordType lbd = event.lbRecord().data();
        lbd |= id->lbRecord().data();
        event.add( LBRecord(lbd) );
        merged = true;
      }
    }
    if (!merged) result.push_back(*id);
  }
  return result;

}

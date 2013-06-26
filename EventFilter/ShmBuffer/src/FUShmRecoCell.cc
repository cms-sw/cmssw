////////////////////////////////////////////////////////////////////////////////
//
// FUShmRecoCell
// -------------
//
//            17/03/2007 Philipp Schieferdecker <philipp.schieferdecker@cern.ch>
////////////////////////////////////////////////////////////////////////////////


#include "EventFilter/ShmBuffer/interface/FUShmRecoCell.h"

#include <iostream>
#include <iomanip>
#include <cstring>

using namespace std;
using namespace evf;


////////////////////////////////////////////////////////////////////////////////
// construction/destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
FUShmRecoCell::FUShmRecoCell(unsigned int payloadSize)
  : payloadSize_(payloadSize)
{
  payloadOffset_=sizeof(FUShmRecoCell);
  void* payloadAddr=(void*)((unsigned long)this+payloadOffset_);
  new (payloadAddr) unsigned char[payloadSize_];
}


//______________________________________________________________________________
FUShmRecoCell::~FUShmRecoCell()
{

}


////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
void FUShmRecoCell::initialize(unsigned int index)
{
  index_=index;
  clear();
}


//______________________________________________________________________________
unsigned char* FUShmRecoCell::payloadAddr() const
{
  unsigned char* result=(unsigned char*)((unsigned long)this+payloadOffset_);
  return result;
}


//______________________________________________________________________________
void FUShmRecoCell::clear()
{
  eventSize_=0;
  rawCellIndex_=0xffffffff;
}


//______________________________________________________________________________
void FUShmRecoCell::writeInitMsg(unsigned int   outModId,
				 unsigned int   fuProcessId,
				 unsigned int   fuGuid,
				 unsigned char *data,
				 unsigned int   dataSize,
				 unsigned int   nExpectedEPs)
{
  if (eventSize_!=0)
    cout<<"FUShmRecoCell::writeInitMsg() WARNING: overwriting data!"<<endl;
  
  if (dataSize>payloadSize_) {
    cout<<"FUShmRecoCell::writeInitMsg() ERROR: data does not fit!"<<endl;
    return;
  }
  
  rawCellIndex_=0xffffffff;
  runNumber_   =0xffffffff;
  evtNumber_   =0xffffffff;
  outModId_    =outModId;
  fuProcessId_ =fuProcessId;
  fuGuid_      =fuGuid;
  type_        =0;
  unsigned char* targetAddr=payloadAddr();
  memcpy(targetAddr,data,dataSize);
  eventSize_=dataSize;
  nExpectedEPs_=nExpectedEPs;
}
				  

//______________________________________________________________________________
void FUShmRecoCell::writeEventData(unsigned int   rawCellIndex,
				   unsigned int   runNumber,
				   unsigned int   evtNumber,
				   unsigned int   outModId,
				   unsigned int   fuProcessId,
				   unsigned int   fuGuid,
				   unsigned char *data,
				   unsigned int   dataSize)
{
  if (eventSize_!=0)
    cout<<"FUShmRecoCell::writeEventData() WARNING: overwriting data!"<<endl;
  
  if (dataSize>payloadSize_) {
    cout<<"FUShmRecoCell::writeEventData() ERROR: data does not fit!"<<endl;
    return;
  }
  
  rawCellIndex_=rawCellIndex;
  runNumber_   =runNumber;
  evtNumber_   =evtNumber;
  outModId_    =outModId;
  fuProcessId_ =fuProcessId;
  fuGuid_      =fuGuid;
  type_        =1;
  unsigned char* targetAddr=payloadAddr();
  memcpy(targetAddr,data,dataSize);
  eventSize_=dataSize;
  nExpectedEPs_=0xffffffff;
}
				  

//______________________________________________________________________________
void FUShmRecoCell::writeErrorEvent(unsigned int   rawCellIndex,
				    unsigned int   runNumber,
				    unsigned int   evtNumber,
				    unsigned int   fuProcessId,
				    unsigned char *data,
				    unsigned int   dataSize)
{
  if (eventSize_!=0)
    cout<<"FUShmRecoCell::writeEventData() WARNING: overwriting data!"<<endl;
  
  if (dataSize>payloadSize_) {
    cout<<"FUShmRecoCell::writeEventData() ERROR: data does not fit!"<<endl;
    return;
  }
  
  rawCellIndex_=rawCellIndex;
  runNumber_   =runNumber;
  evtNumber_   =evtNumber;
  outModId_    =0xffffffff;
  fuProcessId_ =fuProcessId;
  fuGuid_      =0;
  type_        =2;
  unsigned char* targetAddr=payloadAddr();
  memcpy(targetAddr,data,dataSize);
  eventSize_=dataSize;
  nExpectedEPs_=0xffffffff;
}
				  

//______________________________________________________________________________
unsigned int FUShmRecoCell::size(unsigned int payloadSize)
{
  return sizeof(FUShmRecoCell)+sizeof(unsigned char)*payloadSize;
}

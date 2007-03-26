////////////////////////////////////////////////////////////////////////////////
//
// FUShmDqmCell
// ------------
//
//            17/03/2007 Philipp Schieferdecker <philipp.schieferdecker@cern.ch>
////////////////////////////////////////////////////////////////////////////////


#include "EventFilter/ShmBuffer/interface/FUShmDqmCell.h"

#include <iostream>
#include <iomanip>


using namespace std;
using namespace evf;


////////////////////////////////////////////////////////////////////////////////
// construction/destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
FUShmDqmCell::FUShmDqmCell(unsigned int payloadSize)
  : payloadSize_(payloadSize)
{
  payloadOffset_=sizeof(FUShmDqmCell);
  void* payloadAddr=(void*)((unsigned int)this+payloadOffset_);
  new (payloadAddr) unsigned char[payloadSize_];
}


//______________________________________________________________________________
FUShmDqmCell::~FUShmDqmCell()
{

}


////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
void FUShmDqmCell::initialize(unsigned int index)
{
  index_=index;
  clear();
}


//______________________________________________________________________________
unsigned char* FUShmDqmCell::payloadAddr() const
{
  unsigned char* result=(unsigned char*)((unsigned int)this+payloadOffset_);
  return result;
}


//______________________________________________________________________________
void FUShmDqmCell::clear()
{
  eventSize_=0;
}


//______________________________________________________________________________
void FUShmDqmCell::writeData(unsigned int   runNumber,
			     unsigned int   evtAtUpdate,
			     unsigned int   folderId,
			     unsigned char *data,
			     unsigned int   dataSize)
{
  if (eventSize_!=0)
    cout<<"FUShmDqmCell::writeData WARNING: overwriting data!"<<endl;
  
  if (dataSize>payloadSize_) {
    cout<<"FUShmDqmCell::writeData ERROR: data does not fit!"<<endl;
    return;
  }
  
  runNumber_  =runNumber;
  evtAtUpdate_=evtAtUpdate;
  folderId_   =folderId;
  unsigned char* targetAddr=payloadAddr();
  memcpy(targetAddr,data,dataSize);
  eventSize_=dataSize;
}


//______________________________________________________________________________
unsigned int FUShmDqmCell::size(unsigned int payloadSize)
{
  return sizeof(FUShmDqmCell)+sizeof(unsigned char)*payloadSize;
}

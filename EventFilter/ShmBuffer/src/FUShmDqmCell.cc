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
}


//______________________________________________________________________________
unsigned char* FUShmDqmCell::payloadAddr() const
{
  unsigned char* result=(unsigned char*)((unsigned int)this+payloadOffset_);
  return result;
}


//______________________________________________________________________________
void FUShmDqmCell::printState()
{
  switch (state_) {
  case 0 : cout<<"dqm cell "<<index()<<" state: emtpy"  <<endl; return;
  case 1 : cout<<"dqm cell "<<index()<<" state: writing"<<endl; return;
  case 2 : cout<<"dqm cell "<<index()<<" state: written"<<endl; return;
  case 3 : cout<<"dqm cell "<<index()<<" state: sending"<<endl; return;
  case 4 : cout<<"dqm cell "<<index()<<" state: sent"   <<endl; return;
  }
}


//______________________________________________________________________________
void FUShmDqmCell::clear()
{
  setStateEmpty();
  eventSize_=0;
}


//______________________________________________________________________________
void FUShmDqmCell::writeData(unsigned char* data,unsigned int dataSize)
{
  if (eventSize_!=0)
    cout<<"FUShmDqmCell::writeData WARNING: overwriting data!"<<endl;
  
  if (dataSize>payloadSize_) {
    cout<<"FUShmDqmCell::writeData ERROR: data does not fit!"<<endl;
    return;
  }
  
  // result = addr of data to be written *in* the cell
  unsigned char* target=(unsigned char*)((unsigned int)this+payloadOffset_);
  memcpy(target,data,dataSize);
  eventSize_=dataSize;
}


//______________________________________________________________________________
unsigned int FUShmDqmCell::size(unsigned int payloadSize)
{
  return sizeof(FUShmDqmCell)+sizeof(unsigned char)*payloadSize;
}

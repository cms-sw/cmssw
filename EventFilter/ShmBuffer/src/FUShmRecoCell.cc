////////////////////////////////////////////////////////////////////////////////
//
// FUShmRecoCell
// --------------
//
//            17/03/2007 Philipp Schieferdecker <philipp.schieferdecker@cern.ch>
////////////////////////////////////////////////////////////////////////////////


#include "EventFilter/ShmBuffer/interface/FUShmRecoCell.h"

#include <iostream>
#include <iomanip>


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
  void* payloadAddr=(void*)((unsigned int)this+payloadOffset_);
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
}


//______________________________________________________________________________
unsigned char* FUShmRecoCell::payloadAddr() const
{
  unsigned char* result=(unsigned char*)((unsigned int)this+payloadOffset_);
  return result;
}


//______________________________________________________________________________
void FUShmRecoCell::printState()
{
  switch (state_) {
  case 0 : cout<<"reco cell "<<index()<<" state: emtpy"  <<endl; return;
  case 1 : cout<<"reco cell "<<index()<<" state: writing"<<endl; return;
  case 2 : cout<<"reco cell "<<index()<<" state: written"<<endl; return;
  case 3 : cout<<"reco cell "<<index()<<" state: sending"<<endl; return;
  case 4 : cout<<"reco cell "<<index()<<" state: sent"   <<endl; return;
  }
}


//______________________________________________________________________________
void FUShmRecoCell::clear()
{
  setStateEmpty();
  eventSize_=0;
}


//______________________________________________________________________________
void FUShmRecoCell::writeData(unsigned char* data,unsigned int dataSize)
{
  if (eventSize_!=0)
    cout<<"FUShmRecoCell::writeData WARNING: overwriting data!"<<endl;
  
  if (dataSize>payloadSize_) {
    cout<<"FUShmRecoCell::writeData ERROR: data does not fit!"<<endl;
    return;
  }
  
  // result = addr of data to be written *in* the cell
  unsigned char* target=(unsigned char*)((unsigned int)this+payloadOffset_);
  memcpy(target,data,dataSize);
  eventSize_=dataSize;
}
				  

//______________________________________________________________________________
unsigned int FUShmRecoCell::size(unsigned int payloadSize)
{
  return sizeof(FUShmRecoCell)+sizeof(unsigned char)*payloadSize;
}

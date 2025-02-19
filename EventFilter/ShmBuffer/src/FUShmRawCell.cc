////////////////////////////////////////////////////////////////////////////////
//
// FUShmRawCell
// ------------
//
//            09/11/2006 Philipp Schieferdecker <philipp.schieferdecker@cern.ch>
////////////////////////////////////////////////////////////////////////////////


#include "EventFilter/ShmBuffer/interface/FUShmRawCell.h"

#include <iostream>
#include <iomanip>
#include <cstring>

#define NSUPERFRAG_MAX   80
#define NFED_MAX       1024


using namespace std;
using namespace evf;


////////////////////////////////////////////////////////////////////////////////
// construction/destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
FUShmRawCell::FUShmRawCell(unsigned int payloadSize)
  : payloadSize_(payloadSize)
  , nFed_(NFED_MAX)
  , nSuperFrag_(NSUPERFRAG_MAX)
{
  fedSizeOffset_=sizeof(FUShmRawCell);
  unsigned int* fedSizeAddr;
  fedSizeAddr=(unsigned int*)((unsigned long)this+fedSizeOffset_);
  new(fedSizeAddr) unsigned int[nFed_];
  
  fedOffset_=fedSizeOffset_+sizeof(unsigned int)*nFed_;
  unsigned int* fedAddr;
  fedAddr=(unsigned int*)((unsigned long)this+fedOffset_);
  new(fedAddr) unsigned int[nFed_];
  
  superFragSizeOffset_=fedOffset_+sizeof(unsigned int)*nFed_;
  unsigned int* superFragSizeAddr;
  superFragSizeAddr=(unsigned int*)((unsigned long)this+superFragSizeOffset_);
  new(superFragSizeAddr) unsigned int[nSuperFrag_];
  
  superFragOffset_=superFragSizeOffset_+sizeof(unsigned int)*nSuperFrag_;
  unsigned char* superFragAddr;
  superFragAddr=(unsigned char*)((unsigned long)this+superFragOffset_);
  new(superFragAddr) unsigned char[nSuperFrag_];
  
  payloadOffset_=superFragOffset_+sizeof(unsigned int)*nSuperFrag_;
  unsigned char* payloadAddr;
  payloadAddr=(unsigned char*)((unsigned long)this+payloadOffset_);
  new(payloadAddr) unsigned char[payloadSize_];
}


//______________________________________________________________________________
FUShmRawCell::~FUShmRawCell()
{

}


////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
void FUShmRawCell::initialize(unsigned int index)
{
  index_=index;
  lumiSection_=0;
  eventType_=evt::NOP;
}


//______________________________________________________________________________
unsigned int FUShmRawCell::fedSize(unsigned int i) const
{
  if (i>=nFed()) {cout<<"invalid fed index '"<<i<<"'."<<endl; return 0; }
  unsigned int* fedSizeAddr;
  fedSizeAddr=(unsigned int*)((unsigned long)this+fedSizeOffset_);
  fedSizeAddr+=i;
  unsigned int result=*fedSizeAddr;
  return result;
}


//______________________________________________________________________________
unsigned char* FUShmRawCell::fedAddr(unsigned int i) const
{
  if (i>=nFed()) {cout<<"invalid fed index '"<<i<<"'."<<endl; return 0; }
  unsigned int* fedOffsetAddr;
  fedOffsetAddr=(unsigned int*)((unsigned long)this+fedOffset_);
  fedOffsetAddr+=i;
  unsigned int   fedOffset=*fedOffsetAddr;
  unsigned char* result=(unsigned char*)((unsigned long)payloadAddr()+fedOffset);
  return result;
}


//______________________________________________________________________________
unsigned int FUShmRawCell::superFragSize(unsigned int i) const
{
  if (i>=nSuperFrag()) {cout<<"invalid sf index '"<<i<<"'."<<endl; return 0; }
  unsigned int* superFragSizeAddr;
  superFragSizeAddr=(unsigned int*)((unsigned long)this+superFragSizeOffset_);
  superFragSizeAddr+=i;
  unsigned int result=*superFragSizeAddr;
  return result;
}


//______________________________________________________________________________
unsigned char* FUShmRawCell::superFragAddr(unsigned int i) const
{
  if (i>=nSuperFrag()) {cout<<"invalid fed index '"<<i<<"'."<<endl; return 0; }
  unsigned int* superFragOffsetAddr;
  superFragOffsetAddr=(unsigned int*)((unsigned long)this+superFragOffset_);
  superFragOffsetAddr+=i;
  unsigned int   superFragOffset=*superFragOffsetAddr;
  unsigned char* result=(unsigned char*)((unsigned long)payloadAddr()+superFragOffset);
  return result;
}


//______________________________________________________________________________
unsigned char* FUShmRawCell::payloadAddr() const
{
  unsigned char* result=(unsigned char*)((unsigned long)this+payloadOffset_);
  return result;
}


//______________________________________________________________________________
unsigned int FUShmRawCell::eventSize() const
{
  return payloadPosition_;
}


//______________________________________________________________________________
void FUShmRawCell::clear()
{
  fuResourceId_=0xffffffff;
  buResourceId_=0xffffffff;
  lumiSection_ =0;
  eventType_   =evt::NOP;
  unsigned int* fedSizeAddr;
  fedSizeAddr=(unsigned int*)((unsigned long)this+fedSizeOffset_);
  for (unsigned int i=0;i<nFed();i++) *fedSizeAddr++=0;

  unsigned int* superFragSizeAddr;
  superFragSizeAddr=(unsigned int*)((unsigned long)this+superFragSizeOffset_);
  for (unsigned int i=0;i<nSuperFrag();i++) *superFragSizeAddr++=0;

  payloadPosition_=0;
}


//______________________________________________________________________________
void FUShmRawCell::dump() const
{
  for (unsigned int i=0;i<nFed();i++) {
    cout<<"fed "<<i<<": "<<flush;
    unsigned char* addr=fedAddr(i);
    unsigned int   size=fedSize(i);
    cout.fill(0);
    cout<<setiosflags(ios::right);
    for (unsigned int j=0;j<size;j++)
      cout<<setw(2)<<hex<<(int)addr[j]<<dec<<" "<<flush;
    cout<<endl;
  }
}


//______________________________________________________________________________
unsigned int FUShmRawCell::readFed(unsigned int i,
				   unsigned char* buffer) const
{
  unsigned int   size=fedSize(i);
  unsigned char* addr=fedAddr(i);
  memcpy(buffer,addr,size);
  return size;
}


//______________________________________________________________________________
unsigned char* FUShmRawCell::writeData(unsigned char* data,
				       unsigned int   dataSize)
{
  if (payloadPosition_+dataSize>payloadSize_) {
    cout<<"FUShmRawCell::writeData: data to be written does not fit!"<<endl;
    return 0;
  }
  
  // result = addr of data to be written *in* the cell
  unsigned char* result=
    (unsigned char*)((unsigned long)this+payloadOffset_+payloadPosition_);
  memcpy(result,data,dataSize);
  payloadPosition_+=dataSize;
  return result;
}

//______________________________________________________________________________
void FUShmRawCell::setLumiSection(unsigned int ls)
{
  lumiSection_=ls;
}
//______________________________________________________________________________
bool FUShmRawCell::markFed(unsigned int i,
			   unsigned int size,
			   unsigned char* addr)
{
  if (i>=nFed())
    {cout<<"invalid fed index '"<<i<<"'."<<endl; return false; }
  if (addr<payloadAddr())
    { cout<<"invalid fed addr '0x"<<hex<<(unsigned long)addr<<dec<<"'."<<endl; return false; }

  unsigned long offset=(unsigned long)addr-(unsigned long)payloadAddr();

  if (offset>=payloadSize())
    { cout<<"invalid fed addr '0x"<<hex<<(unsigned long)addr<<dec<<"'."<<endl; return false; }

  unsigned int* fedSizeAddr;
  fedSizeAddr=(unsigned int*)((unsigned long)this+fedSizeOffset_);
  fedSizeAddr+=i;
  *fedSizeAddr=size;

  unsigned int* fedAddr;
  fedAddr=(unsigned int*)((unsigned long)this+fedOffset_);
  fedAddr+=i;
  *fedAddr=offset;

  return true;
}


//______________________________________________________________________________
bool FUShmRawCell::markSuperFrag(unsigned int i,
				 unsigned int size,
				 unsigned char* addr)
{
  if (i>=nSuperFrag())
    {cout<<"invalid sf index '"<<i<<"'."<<endl; return false; }
  if (addr<payloadAddr())
    {cout<<"invalid sf addr '0x"<<hex<<(unsigned long)addr<<dec<<"'."<<endl;return false;}

  unsigned int offset=(unsigned long)addr-(unsigned long)payloadAddr();

  if (offset>=payloadSize())
    {cout<<"invalid sf addr '0x"<<hex<<(unsigned long)addr<<dec<<"'."<<endl;return false;}

  unsigned int* superFragSizeAddr;
  superFragSizeAddr=(unsigned int*)((unsigned long)this+superFragSizeOffset_);
  superFragSizeAddr+=i;
  *superFragSizeAddr=size;

  unsigned int* superFragAddr;
  superFragAddr=(unsigned int*)((unsigned long)this+superFragOffset_);
  superFragAddr+=i;
  *superFragAddr=offset;

  return true;
}
				  

//______________________________________________________________________________
unsigned int FUShmRawCell::size(unsigned int payloadSize)
{
  return 
    sizeof(FUShmRawCell)+
    sizeof(unsigned int)*2*(NFED_MAX+NSUPERFRAG_MAX)+
    sizeof(unsigned char)*payloadSize;
}

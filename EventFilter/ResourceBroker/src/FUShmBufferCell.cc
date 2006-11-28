////////////////////////////////////////////////////////////////////////////////
//
// FUShmBufferCell
// ---------------
//
//            09/11/2006 Philipp Schieferdecker <philipp.schieferdecker@cern.ch>
////////////////////////////////////////////////////////////////////////////////


#include "EventFilter/ResourceBroker/interface/FUShmBufferCell.h"

#include <iomanip>


using namespace std;
using namespace evf;


////////////////////////////////////////////////////////////////////////////////
// construction/destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
FUShmBufferCell::FUShmBufferCell(unsigned int index,
				 unsigned int bufferSize,
				 unsigned int nFed,
				 unsigned int nSuperFrag,
				 bool         ownsMemory)
  : ownsMemory_(ownsMemory)
  , fuResourceId_(index)
  , bufferSize_(bufferSize)
  , nFed_(nFed)
  , nSuperFrag_(nSuperFrag)
{
  if (this->ownsMemory()) {
    unsigned int* fedSizeArray=new unsigned int[nFed_];
    fedSizeOffset_=(unsigned int)fedSizeArray-(unsigned int)this;
    unsigned int* fedArray=new unsigned int[nFed_];
    fedOffset_=(unsigned int)fedArray-(unsigned int)this;
    unsigned int* superFragSizeArray=new unsigned int[nSuperFrag_];
    superFragSizeOffset_=(unsigned int)superFragSizeArray-(unsigned int)this;
    unsigned int* superFragArray=new unsigned int[nSuperFrag_];
    superFragOffset_=(unsigned int)superFragArray-(unsigned int)this;
    unsigned char* buffer=new unsigned char[bufferSize_];
    bufferOffset_=(unsigned int)buffer-(unsigned int)this;
  }
  else {
    fedSizeOffset_=sizeof(FUShmBufferCell);
    unsigned int* fedSizeAddr;
    fedSizeAddr=(unsigned int*)((unsigned int)this+fedSizeOffset_);
    new(fedSizeAddr) unsigned int[nFed_];
   
    fedOffset_=fedSizeOffset_+sizeof(unsigned int)*nFed_;
    unsigned int* fedAddr;
    fedAddr=(unsigned int*)((unsigned int)this+fedOffset_);
    new(fedAddr) unsigned int[nFed_];
   
    superFragSizeOffset_=fedOffset_+sizeof(unsigned int)*nFed_;
    unsigned int* superFragSizeAddr;
    superFragSizeAddr=(unsigned int*)((unsigned int)this+superFragSizeOffset_);
    new(superFragSizeAddr) unsigned int[nSuperFrag_];
    
    superFragOffset_=superFragSizeOffset_+sizeof(unsigned int)*nSuperFrag_;
    unsigned char* superFragAddr;
    superFragAddr=(unsigned char*)((unsigned int)this+superFragOffset_);
    new(superFragAddr) unsigned char[nSuperFrag_];

    bufferOffset_=superFragOffset_+sizeof(unsigned int)*nSuperFrag_;
    unsigned char* bufferAddr;
    bufferAddr=(unsigned char*)((unsigned int)this+bufferOffset_);
    new(bufferAddr) unsigned char[bufferSize_];
  }
}


//______________________________________________________________________________
FUShmBufferCell::~FUShmBufferCell()
{
  if (ownsMemory()) {
    unsigned int* fedSizeAddr;
    fedSizeAddr=(unsigned int*)((unsigned int)this+fedSizeOffset_);
    delete [] fedSizeAddr;
    unsigned int* fedAddr;
    fedAddr=(unsigned int*)((unsigned int)this+fedOffset_);
    delete [] fedAddr;
    unsigned int* superFragSizeAddr;
    superFragSizeAddr=(unsigned int*)((unsigned int)this+superFragSizeOffset_);
    delete [] superFragSizeAddr;
    unsigned int* superFragAddr;
    superFragAddr=(unsigned int*)((unsigned int)this+superFragOffset_);
    delete [] superFragAddr;
    unsigned char* bufferAddr;
    bufferAddr=(unsigned char*)((unsigned int)this+bufferOffset_);
    delete [] bufferAddr;
  }
}


////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
unsigned int FUShmBufferCell::fedSize(unsigned int i) const
{
  if (i>=nFed()) {cout<<"invalid fed index '"<<i<<"'."<<endl; return 0; }
  unsigned int* fedSizeAddr;
  fedSizeAddr=(unsigned int*)((unsigned int)this+fedSizeOffset_);
  fedSizeAddr+=i;
  unsigned int result=*fedSizeAddr;
  return result;
}


//______________________________________________________________________________
unsigned char* FUShmBufferCell::fedAddr(unsigned int i) const
{
  if (i>=nFed()) {cout<<"invalid fed index '"<<i<<"'."<<endl; return 0; }
  unsigned int* fedOffsetAddr;
  fedOffsetAddr=(unsigned int*)((unsigned int)this+fedOffset_);
  fedOffsetAddr+=i;
  unsigned int   fedOffset=*fedOffsetAddr;
  unsigned char* result=(unsigned char*)((unsigned int)bufferAddr()+fedOffset);
  return result;
}


//______________________________________________________________________________
unsigned int FUShmBufferCell::superFragSize(unsigned int i) const
{
  if (i>=nSuperFrag()) {cout<<"invalid sf index '"<<i<<"'."<<endl; return 0; }
  unsigned int* superFragSizeAddr;
  superFragSizeAddr=(unsigned int*)((unsigned int)this+superFragSizeOffset_);
  superFragSizeAddr+=i;
  unsigned int result=*superFragSizeAddr;
  return result;
}


//______________________________________________________________________________
unsigned char* FUShmBufferCell::superFragAddr(unsigned int i) const
{
  if (i>=nSuperFrag()) {cout<<"invalid fed index '"<<i<<"'."<<endl; return 0; }
  unsigned int* superFragOffsetAddr;
  superFragOffsetAddr=(unsigned int*)((unsigned int)this+superFragOffset_);
  superFragOffsetAddr+=i;
  unsigned int   superFragOffset=*superFragOffsetAddr;
  unsigned char* result=(unsigned char*)((unsigned int)bufferAddr()+superFragOffset);
  return result;
}


//______________________________________________________________________________
unsigned char* FUShmBufferCell::bufferAddr() const
{
  unsigned char* result=(unsigned char*)((unsigned int)this+bufferOffset_);
  return result;
}


//______________________________________________________________________________
unsigned int FUShmBufferCell::eventSize() const
{
  unsigned int result(0);
  for (unsigned int i=0;i<nSuperFrag();i++) result+=superFragSize(i);

  unsigned int result2=bufferPosition_;

  assert(result==result2);
    
  return result;
}


//______________________________________________________________________________
void FUShmBufferCell::clear()
{
  setStateEmpty();
  
  unsigned int* fedSizeAddr;
  fedSizeAddr=(unsigned int*)((unsigned int)this+fedSizeOffset_);
  for (unsigned int i=0;i<nFed();i++) *fedSizeAddr++=0;

  /// unnecessary, isn't it?
  //unsigned int* fedAddr;
  //fedAddr=(unsigned int*)((unsigned int)this+fedOffset_);
  //for (unsigned int i=0;i<nFed();i++) *fedAddr++=0;

  unsigned int* superFragSizeAddr;
  superFragSizeAddr=(unsigned int*)((unsigned int)this+superFragSizeOffset_);
  for (unsigned int i=0;i<nSuperFrag();i++) *superFragSizeAddr++=0;

  /// unnecessary, isn't it?
  //unsigned int* superFragAddr;
  //superFragAddr=(unsigned int*)((unsigned int)this+superFragOffset_);
  //for (unsigned int i=0;i<nSuperFrag();i++) *superFragAddr++=0;

  bufferPosition_=0;
}


//______________________________________________________________________________
void FUShmBufferCell::print(int verbose) const
{
  cout<<"FUShmBufferCell:"
      <<" fuResourceId="<<fuResourceId()
      <<" buResourceId="<<buResourceId()
      <<" evtNumber="<<evtNumber()
      <<" this=0x"<<hex<<(int)this<<dec
      <<" ownsMemory="<<ownsMemory()
      <<endl
      <<"                "
      <<" bufferSize="<<bufferSize()
      <<" nFed="<<nFed()
      <<" nSuperFrag="<<nSuperFrag()
      <<" eventSize="<<eventSize()
      <<endl;
  if (verbose>0) {
    for (unsigned int i=0;i<nFed();i++)
      cout<<" "<<i<<". fed:"
	  <<" size="<<fedSize(i)
	  <<" addr0x="<<hex<<(int)fedAddr(i)<<dec
	  <<endl;
  }
}


//______________________________________________________________________________
void FUShmBufferCell::dump() const
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
unsigned int FUShmBufferCell::readFed(unsigned int i,
				      unsigned char* buffer) const
{
  unsigned int   size=fedSize(i);
  unsigned char* addr=fedAddr(i);
  memcpy(buffer,addr,size);
  return size;
}


//______________________________________________________________________________
unsigned char* FUShmBufferCell::writeData(unsigned char* data,
					  unsigned int   dataSize)
{
  if (bufferPosition_+dataSize>bufferSize_) {
    cout<<"FUShmBufferCell::writeData: data to be written does not fit!"<<endl;
    return 0;
  }
  
  // result = addr of data to be written *in* the cell
  unsigned char* result=
    (unsigned char*)((unsigned int)this+bufferOffset_+bufferPosition_);
  memcpy(result,data,dataSize);
  bufferPosition_+=dataSize;
  return result;
}


//______________________________________________________________________________
bool FUShmBufferCell::markFed(unsigned int i,
			      unsigned int size,
			      unsigned char* addr)
{
  if (i>=nFed())
    {cout<<"invalid fed index '"<<i<<"'."<<endl; return false; }
  if (addr<bufferAddr())
    { cout<<"invalid fed addr '0x"<<hex<<(int)addr<<dec<<"'."<<endl; return false; }

  unsigned int offset=(unsigned int)addr-(unsigned int)bufferAddr();

  if (offset>=bufferSize())
    { cout<<"invalid fed addr '0x"<<hex<<(int)addr<<dec<<"'."<<endl; return false; }

  unsigned int* fedSizeAddr;
  fedSizeAddr=(unsigned int*)((unsigned int)this+fedSizeOffset_);
  fedSizeAddr+=i;
  *fedSizeAddr=size;

  unsigned int* fedAddr;
  fedAddr=(unsigned int*)((unsigned int)this+fedOffset_);
  fedAddr+=i;
  *fedAddr=offset;

  return true;
}


//______________________________________________________________________________
bool FUShmBufferCell::markSuperFrag(unsigned int i,
				    unsigned int size,
				    unsigned char* addr)
{
  if (i>=nSuperFrag())
    {cout<<"invalid sf index '"<<i<<"'."<<endl; return false; }
  if (addr<bufferAddr())
    {cout<<"invalid sf addr '0x"<<hex<<(int)addr<<dec<<"'."<<endl;return false;}

  unsigned int offset=(unsigned int)addr-(unsigned int)bufferAddr();

  if (offset>=bufferSize())
    {cout<<"invalid sf addr '0x"<<hex<<(int)addr<<dec<<"'."<<endl;return false;}

  unsigned int* superFragSizeAddr;
  superFragSizeAddr=(unsigned int*)((unsigned int)this+superFragSizeOffset_);
  superFragSizeAddr+=i;
  *superFragSizeAddr=size;

  unsigned int* superFragAddr;
  superFragAddr=(unsigned int*)((unsigned int)this+superFragOffset_);
  superFragAddr+=i;
  *superFragAddr=offset;

  return true;
}
				  

//______________________________________________________________________________
unsigned int FUShmBufferCell::size(unsigned int bufferSize,
				   unsigned int nFed,
				   unsigned int nSuperFrag)

{
  return 
    sizeof(FUShmBufferCell)+
    sizeof(unsigned int)*2*(nFed+nSuperFrag)+
    sizeof(unsigned char)*bufferSize;
}

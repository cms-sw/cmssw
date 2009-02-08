////////////////////////////////////////////////////////////////////////////////
//
// BUEvent
// -------
//
//            03/26/2007 Philipp Schieferdecker <philipp.schieferdecker@cern.ch>
////////////////////////////////////////////////////////////////////////////////


#include "EventFilter/AutoBU/interface/BUEvent.h"
#include <assert.h>
#include "FWCore/Utilities/interface/CRC16.h"

#include "interface/shared/fed_header.h"
#include "interface/shared/fed_trailer.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cstring>

using namespace std;
using namespace evf;



////////////////////////////////////////////////////////////////////////////////
// initialize static member data 
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
bool BUEvent::computeCrc_=true;


////////////////////////////////////////////////////////////////////////////////
// construction/destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
BUEvent::BUEvent(unsigned int buResourceId,unsigned int bufferSize)
  : buResourceId_(buResourceId)
  , evtNumber_(0xffffffff)
  , evtSize_(0)
  , bufferSize_(bufferSize)
  , nFed_(0)
  , fedId_(0)
  , fedPos_(0)
  , fedSize_(0)
  , buffer_(0)
{
  fedId_  = new unsigned int[1024];
  fedPos_ = new unsigned int[1024];
  fedSize_= new unsigned int[1024];
  buffer_ = new unsigned char[bufferSize];
}


//______________________________________________________________________________
BUEvent::~BUEvent()
{
  if (0!=fedId_)   delete [] fedId_;
  if (0!=fedPos_)  delete [] fedPos_;
  if (0!=fedSize_) delete [] fedSize_;
  if (0!=buffer_)  delete [] buffer_;
}


////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
void BUEvent::initialize(unsigned int evtNumber)
 {
   evtNumber_=evtNumber & 0xFFFFFF; // 24 bits only available in the FED headers
   evtSize_=0;
   nFed_=0;
 }


//______________________________________________________________________________
bool BUEvent::writeFed(unsigned int id,unsigned char* data,unsigned int size)
{
  if (evtSize_+size > bufferSize_) {
    cout<<"BUEvent::writeFed() ERROR: buffer overflow."<<endl;
    return false;
  }
  
  if (nFed_==1024) {
    cout<<"BUEvent::writeFed() ERROR: too many feds (max=1024)."<<endl;
    return false;
  }
  
  fedId_[nFed_]  =id;
  fedPos_[nFed_] =evtSize_;
  fedSize_[nFed_]=size;
  if (0!=data) memcpy(fedAddr(nFed_),data,size);
  ++nFed_;
  evtSize_+=size;
  return true;
}


//______________________________________________________________________________
bool BUEvent::writeFedHeader(unsigned int i)
{
  if (i>=nFed_) {
    cout<<"BUEvent::writeFedHeader() ERROR: invalid fed index '"<<i<<"'."<<endl;
    return false;
  }
  
  fedh_t *fedHeader=(fedh_t*)fedAddr(i);
  fedHeader->eventid =evtNumber();
  fedHeader->eventid|=0x50000000;
  fedHeader->sourceid=(fedId(i) << 8) & FED_SOID_MASK;
  
  return true;
}


//______________________________________________________________________________
bool BUEvent::writeFedTrailer(unsigned int i)
{
  if (i>=nFed_) {
    cout<<"BUEvent::writeFedTrailer() ERROR: invalid fed index '"<<i<<"'."<<endl;
    return false;
  }
  
  fedt_t *fedTrailer=(fedt_t*)(fedAddr(i)+fedSize(i)-sizeof(fedt_t));
  fedTrailer->eventsize =fedSize(i);
  fedTrailer->eventsize/=8; //wc in fed trailer in 64bit words
  fedTrailer->eventsize|=0xa0000000;
  fedTrailer->conscheck =0x0;
  
  if (BUEvent::computeCrc()) {
    unsigned short crc=evf::compute_crc(fedAddr(i),fedSize(i));
    fedTrailer->conscheck=(crc<<FED_CRCS_SHIFT);
  }

  return true;
}


//______________________________________________________________________________
unsigned char* BUEvent::fedAddr(unsigned int i) const
{
  return (buffer_+fedPos_[i]);
}


//________________________________________________________________________________
void BUEvent::dump()
{
  ostringstream oss; oss<<"/tmp/autobu_evt"<<evtNumber()<<".dump";
  ofstream fout(oss.str().c_str());
  fout.fill('0');
  
  fout<<"#\n# evt "<<evtNumber()<<"\n#\n"<<endl;
  for (unsigned int i=0;i<nFed();i++) {
    if (fedSize(i)==0) continue;
    fout<<"# fedid "<<fedId(i)<<endl;
    unsigned char* addr=fedAddr(i);
    for (unsigned int j=0;j<fedSize(i);j++) {
      fout<<setiosflags(ios::right)<<setw(2)<<hex<<(int)(*addr)<<dec;
      if ((j+1)%8) fout<<" "; else fout<<endl;
      ++addr;
    }
    fout<<endl;
  }
  fout.close();
}

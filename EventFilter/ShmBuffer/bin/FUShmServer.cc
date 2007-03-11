////////////////////////////////////////////////////////////////////////////////
//
// FUShmServer
// -----------
//
//            17/11/2006 Philipp Schieferdecker <philipp.schieferdecker@cern.ch>
////////////////////////////////////////////////////////////////////////////////


#include "EventFilter/ShmBuffer/bin/FUShmServer.h"


#include <iostream>


using namespace std;
using namespace evf;


////////////////////////////////////////////////////////////////////////////////
// construction/destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
FUShmServer::FUShmServer(FUShmBuffer* buffer)
  : buffer_(buffer)
{
  
}


//______________________________________________________________________________
FUShmServer::~FUShmServer()
{
  
}


////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
unsigned int FUShmServer::writeNext(unsigned char *data,
				    unsigned int   nFed,
				    unsigned int  *fedSize)
{
  // decrement writer sem
  buffer_->waitWriterSem();

  // lock buffer
  buffer_->lock();
  
  // write and advance
  FUShmBufferCell* cell =buffer_->currentWriterCell();
  cout<<"STATE "<<flush;
  if (cell->isEmpty())     cout<<"empty"     <<endl;
  if (cell->isWritten())   cout<<"written"   <<endl;
  if (cell->isProcessing())cout<<"processing"<<endl;
  if (cell->isProcessed()) cout<<"processed" <<endl;
  if (cell->isDead())      cout<<"dead" <<endl;
  
  if (!cell->isEmpty()) {
    if (cell->isProcessed())
      cout<<"DISCARD "<<cell->index()<<endl;
    else if (cell->isDead())
      cout<<"dead cell "<<cell->index()<<", HANDLE&DISCARD"<<endl;
    else{
      cout<<"ERROR: unexpected state of cell "<<cell->index()<<endl;
      cell->dump();
    }
  }
  
  // unlock buffer
  buffer_->unlock();
  
  // write data
  cell->clear();
  unsigned int dataSize(0);
  for (unsigned int i=0;i<nFed;i++) dataSize+=fedSize[i];
  unsigned int   iCell         =cell->index();
  unsigned char *cellBufferAddr=cell->writeData(data,dataSize);
  
  if (0!=cellBufferAddr) {
    // marks feds
    unsigned int fedOffset(0);
    for (unsigned int i=0;i<nFed;i++) {
      unsigned char* fedAddr=cellBufferAddr+fedOffset;
      cell->markFed(i,fedSize[i],fedAddr);
      fedOffset+=fedSize[i];
    }
    
    // set cell state to 'written'
    cell->setStateWritten();
    
    // increment the reader sem
    buffer_->postReaderSem();
  }
  
  return iCell;
}
  

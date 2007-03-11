////////////////////////////////////////////////////////////////////////////////
//
// FUShmClient
// -----------
//
//            17/11/2006 Philipp Schieferdecker <philipp.schieferdecker@cern.ch>
////////////////////////////////////////////////////////////////////////////////


#include "EventFilter/ShmBuffer/bin/FUShmClient.h"


#include <iostream>
#include <cstdlib>   // rand()
#include <unistd.h> // sleep


using namespace std;
using namespace evf;


double get_rnd();


////////////////////////////////////////////////////////////////////////////////
// construction/destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
FUShmClient::FUShmClient(FUShmBuffer* buffer)
  : buffer_(buffer)
  , crashPrb_(0.01)
  , sleep_(0)
{
  
}


//______________________________________________________________________________
FUShmClient::~FUShmClient()
{

}


////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
unsigned int FUShmClient::readNext(vector<vector<unsigned char> >& feds)
{
  // decrement the client sem
  buffer_->waitReaderSem();
  
  // lock buffer
  buffer_->lock();
  
  // read&advance
  FUShmBufferCell* cell =buffer_->currentReaderCell();
  
  // this would be seriously troubling!!
  while (!cell->isWritten()) {
    cout<<"ERROR: unexpected state of cell "<<cell->index()<<endl;
    buffer_->sem_print();
    buffer_->print();
    cell=buffer_->currentReaderCell();

  }
  
  // set state of the cell to 'Processing'
  cell->setStateProcessing();
  
  // unlock buffer
  buffer_->unlock();

  // read data
  unsigned int iCell=cell->index();
  unsigned int nFed =cell->nFed();
  feds.resize(nFed);
  for (unsigned int i=0;i<nFed;i++) {
    unsigned int   fedSize =cell->fedSize(i);
    feds[i].resize(fedSize);  
    unsigned char *destAddr=(unsigned char*)&(feds[i][0]);
    cell->readFed(i,destAddr);
  }
  
  // sleep
  if (sleep_>0.0) {
    cout<<"PROCESSING cell "<<cell->index()<<" ... "<<flush;
    sleep(sleep_);
    cout<<"DONE"<<endl;
  }
  
  //crash
  if (get_rnd()<crashPrb()) {
    cout<<"FUShmClient::readNext(): CRASH! cell->index()="<<cell->index()<<endl;
    exit(1);
  }

  // set the state of the cell to 'processed'
  cell->setStateProcessed();
  
  // increment the writer sem
  buffer_->postWriterSem();
  
  return iCell;
}


//______________________________________________________________________________
double get_rnd()
{
  double result=rand()/(double)RAND_MAX;
  return result;
}

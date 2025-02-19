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
#include <unistd.h>  // sleep


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
  FUShmRawCell* cell=buffer_->rawCellToRead();
  
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
  buffer_->finishReadingRawCell(cell);

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

  buffer_->scheduleRawCellForDiscard(iCell);
  cell=buffer_->rawCellToDiscard();
  buffer_->discardRawCell(cell);

  return iCell;
}


//______________________________________________________________________________
double get_rnd()
{
  double result=rand()/(double)RAND_MAX;
  return result;
}

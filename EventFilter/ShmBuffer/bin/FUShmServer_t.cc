////////////////////////////////////////////////////////////////////////////////
//
// FUShmServer_t
// -------------
//
//            17/11/2006 Philipp Schieferdecker <philipp.schieferdecker@cern.ch>
////////////////////////////////////////////////////////////////////////////////


#include "EventFilter/ShmBuffer/interface/FUShmBuffer.h"
#include "EventFilter/ShmBuffer/bin/FUShmServer.h"

#include <iostream>
#include <cstdlib>   // rand()
#include <sstream>


using namespace std;
using namespace evf;


//______________________________________________________________________________
int main(int argc,char**argv)
{
  bool         segmentationMode=false;
  unsigned int nCells          =32;
  unsigned int nFed            = 4;
  unsigned int bytesPerFed     =10;
  unsigned int cellBufferSize  =nFed*bytesPerFed;
  
  if (argc>1) { stringstream ss; ss<<argv[1]; ss>>segmentationMode; }
  
  cout<<" FUShmServer_t:"
      <<" segmentationMode="<<segmentationMode
      <<" nCells="<<nCells
      <<" nFed="<<nFed
      <<" bytesPerFed="<<bytesPerFed
      <<" cellBufferSize="<<cellBufferSize<<endl<<endl;
  
  FUShmBuffer* buffer=FUShmBuffer::createShmBuffer(segmentationMode,
						   nCells,0,0,
						   cellBufferSize,0,0);
  if (0==buffer) return 1;
  FUShmServer* server=new FUShmServer(buffer);
  
  
  // the fake data written to shared memory
  unsigned char* data         =new unsigned char[cellBufferSize];
  unsigned int*  fedSize      =new unsigned int[nFed];
  for (unsigned int i=0;i<nFed;i++) fedSize[i]=bytesPerFed;
  
  
  // server loop
  while(1) {
    
    // generate data
    for (unsigned int i=0;i<cellBufferSize;i++) {
      unsigned int rnd=rand();
      double tmp=rnd/(double)RAND_MAX*255;
      rnd=(unsigned int)tmp;
      data[i]=(unsigned char)rnd;
    }
    
    unsigned int iCell=server->writeNext(data,nFed,fedSize);
    cout<<"WROTE at index "<<iCell<<endl;
  }
  
  return 0;
}

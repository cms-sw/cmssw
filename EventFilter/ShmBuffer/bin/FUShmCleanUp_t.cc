////////////////////////////////////////////////////////////////////////////////
//
// FUShmCleanUp_t
// --------------
//
//            19/11/2006 Philipp Schieferdecker <philipp.schieferdecker@cern.ch>
////////////////////////////////////////////////////////////////////////////////


#include "EventFilter/ShmBuffer/interface/FUShmBuffer.h"

#include <iostream>


using namespace std;
using namespace evf;


//______________________________________________________________________________
int main(int argc,char** argv)
{
  if (FUShmBuffer::releaseSharedMemory())
    cout<<"SHARED MEMORY RELEASED SUCCESSFULLY."<<endl;
  else
    cout<<"NO SHARED MEMORY RELEASED!"<<endl;
  
  return 0;
}

////////////////////////////////////////////////////////////////////////////////
//
// FUShmClient_t
// -------------
//
//            17/11/2006 Philipp Schieferdecker <philipp.schieferdecker@cern.ch>
////////////////////////////////////////////////////////////////////////////////


#include "EventFilter/ShmBuffer/interface/FUShmBuffer.h"
#include "EventFilter/ShmBuffer/bin/FUShmClient.h"


#include <iostream>
#include <iomanip>
#include <sstream>
#include <unistd.h>


using namespace std;
using namespace evf;


//______________________________________________________________________________
void print_fed(unsigned int iFed,const vector<unsigned char>& fedData);


//______________________________________________________________________________
int main(int argc,char**argv)
{
  // set parameters
  bool   waitForKey(false);if (argc>1) {stringstream ss;ss<<argv[1];ss>>waitForKey;}
  double crashPrb(0.0);    if (argc>2) {stringstream ss;ss<<argv[2];ss>>crashPrb;  }
  int    sleepTime(0);     if (argc>3) {stringstream ss;ss<<argv[3];ss>>sleepTime; }

  // get shared memory buffer and instantiate client
  FUShmBuffer* buffer=FUShmBuffer::getShmBuffer(); if (0==buffer) return 1;
  FUShmClient* client=new FUShmClient(buffer);
  client->setCrashPrb(crashPrb);
  client->setSleep(sleepTime);
  
  // the structure to hold the read fed data
  vector<vector<unsigned char> > fedData;
  
  // client loop
  while(1) {
    
    // wait for the user to press a key
    if (waitForKey) {
      cout<<"press key+<RETURN> to read next: "<<flush; char val; cin>>val;
      cout<<endl;
    }
    
    // read next event
    unsigned int iCell=client->readNext(fedData);
    cout<<"READ at index "<<iCell<<endl;

    // print content
    for(unsigned int i=0;i<4 /*fedData.size()*/;i++) print_fed(i,fedData[i]);
    cout<<endl;
    
  }
  
  return 0;
}


//______________________________________________________________________________
void print_fed(unsigned int iFed,const vector<unsigned char>& fedData)
{
  cout<<"fed "<<iFed<<": "<<flush;
  vector<unsigned char>::const_iterator it;
  cout.fill('0');
  for (it=fedData.begin();it!=fedData.end();++it)
    cout<<setiosflags(ios::right)<<setw(2)<<hex<<(int)(*it)<<dec<<" ";
  cout<<endl;
}

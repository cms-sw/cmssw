#include <vector>
#include <TFile.h>
using namespace std;

#if defined(__CINT__) && !defined(__MAKECINT__)
class loadFWLite {
   public:
      loadFWLite() {
         gSystem->Load("libFWCoreFWLite");
         AutoLibraryLoader::enable();
      }
};

static loadFWLite lfw;
#endif

#include "DataFormats/FWLite/interface/Handle.h"

#if !defined(__CINT__) && !defined(__MAKECINT__)
#include "DataFormats/TestObjects/interface/ThingCollection.h"
#endif

void event_looping_cint()
{
TFile f("good.root");
fwlite::Event e(&f);

int i =0;
int returnValue = 0;
for( ;e.isValid();++e,++i) {
  fwlite::Handle<vector<edmtest::Thing> > pThing;
  //pThing.getByLabel(e,"Thing","","TEST"); //WORKS
  pThing.getByLabel(e,"Thing");
  
  for(int i=0; i!=pThing.ref().size();++i) {
    cout <<pThing.ref().at(i).a<< " ";
  }
  cout << endl;
}  
if (i==0) {
  cout <<"First loop failed!"<<endl;
  returnValue = 1;
}
e.toBegin();

i=0;
for( ;e;++e,++i) { 
}

if (i==0) {
  cout <<"Second loop failed!"<<endl;
  returnValue = 1;
}

i=0;
for(e.toBegin(); !e.atEnd();++e,++i) {
   fwlite::Handle<vector<edmtest::Thing> > pThing;
   //pThing.getByLabel(e,"Thing","","TEST"); //WORKS
   pThing.getByLabel(e,"Thing");
   
   for(int i=0; i!=pThing.ref().size();++i) {
      cout <<pThing.ref().at(i).a<<" ";
   }
   cout << endl;

   //DOES NOT WORK
   //for(vector<edmtest::Thing>::const_iterator it = pThing.data()->begin(); it != pThing.data()->end();++it) {
   //   cout <<(*it).a<<endl;
   //}
}
if (i==0) {
  cout <<"Third loop failed!"<<endl;
  returnValue = 1;
}
exit(returnValue);
}

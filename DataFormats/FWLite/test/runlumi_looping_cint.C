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

void runlumi_looping_cint()
{
TFile f("good.root");
fwlite::Run r(&f);
fwlite::LuminosityBlock l(&f);

int i =0;
int returnValue = 0;
for( ;r.isValid();++r,++i) {
  cout << r.run() << endl;
}  
if (i==0) {
  cout <<"First run loop failed!"<<endl;
  returnValue = 1;
}

int i =0;
int returnValue = 0;
for( ;l.isValid();++l,++i) {
  cout << l.id().run() << " " << l.id().luminosityBlock() << endl;
}  
if (i==0) {
  cout <<"First lumi loop failed!"<<endl;
  returnValue = 1;
}


r.toBegin();
i=0;
for( ;r;++r,++i) { 
}
if (i==0) {
  cout <<"Second run loop failed!"<<endl;
  returnValue = 1;
}

l.toBegin();
i=0;
for( ;l;++l,++i) { 
}
if (i==0) {
  cout <<"Second lumi loop failed!"<<endl;
  returnValue = 1;
}

i=0;
for(r.toBegin(); !r.atEnd();++r,++i) {
  cout << r.run() << endl;
}
if (i==0) {
  cout <<"Third run loop failed!"<<endl;
  returnValue = 1;
}

i=0;
for(l.toBegin(); !l.atEnd();++l,++i) {
  cout << l.id().run() << " " << l.id().luminosityBlock() << endl;
}
if (i==0) {
  cout <<"Third lumi loop failed!"<<endl;
  returnValue = 1;
}

exit(returnValue);
}

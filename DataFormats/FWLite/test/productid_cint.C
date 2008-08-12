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

#if !defined(__CINT__) && !defined(__MAKECINT__)
#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/TestObjects/interface/ThingCollection.h"
#endif

#include "DataFormats/FWLite/interface/Handle.h"

void productid_cint()
{
  TFile f1("prodmerge.root");
  fwlite::Event ev(&f1);
  
  for( ev.toBegin(); ! ev.atEnd(); ++ev) {
    edm::EventID id = ev.id();
    cout << "Run " << id.run() << " event " << id.event() << endl;
    fwlite::Handle<vector<edmtest::Thing> > pThing;
    pThing.getByLabel(ev,"Thing");
    for(unsigned int i=0; i!=pThing.ref().size();++i) {
      cout <<pThing.ref().at(i).a<<endl;
    }
  }
}

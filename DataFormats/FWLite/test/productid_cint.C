#include <vector>
#include <TFile.h>
using namespace std;

#if defined(__CINT__) && !defined(__MAKECINT__)
class loadFWLite {
   public:
      loadFWLite() {
         gSystem->Load("libFWCoreFWLite");
         FWLiteEnabler::enable();
      }
};

static loadFWLite lfw;
#endif

#include "DataFormats/FWLite/interface/Handle.h"

#if !defined(__CINT__) && !defined(__MAKECINT__)
#include "DataFormats/TestObjects/interface/ThingCollection.h"
#endif

void productid_cint()
{
  TFile f1("prodmerge.root");
  fwlite::Event ev(&f1);
  fwlite::Handle<vector<edmtest::Thing> > pThing;
  fwlite::Handle<vector<edmtest::OtherThing> > oThing;

  // test that getProcessHistory() and getBranchName() work before getting the first event
  const vector<string>& hist = ev.getProcessHistory();
  for (unsigned int i=0; i != hist.size(); ++i) {
    cout << hist.at(i) << " " << pThing.getBranchNameFor(ev,"Thing","",hist.at(i).c_str()) << endl;
  }
  cout << "No such thing: "<< pThing.getBranchNameFor(ev,"NoSuchThing") << endl;
  
  for (ev.toBegin(); ! ev.atEnd(); ++ev) {
    edm::EventID id = ev.id();
    cout << "Run " << id.run() << " event " << id.event() << endl;
    pThing.getByLabel(ev,"Thing","","TEST");
    cout << pThing.getBranchNameFor(ev,"Thing","","TEST") << " ";
    for (unsigned int i=0; i != pThing.ref().size(); ++i) {
      cout <<pThing.ref().at(i).a<<" ";
    }
    cout << endl;
    pThing.getByLabel(ev,"Thing");
    cout << pThing.getBranchNameFor(ev,"Thing") << " ";
    for (unsigned int i=0; i != pThing.ref().size(); ++i) {
      cout <<pThing.ref().at(i).a<<" ";
    }
    cout << endl;
    oThing.getByLabel(ev,"OtherThing","","FOO");
    cout << "Nonexistent other thing valid: " << oThing.isValid() << " failedToGet: " << oThing.failedToGet() << endl;
  }

  bool t = ev.to(1,2);
  edm::EventID id = ev.id();
  pThing.getByLabel(ev,"Thing");
  cout << t << " Run " << id.run() << " event " << id.event() << " " << pThing.ref().at(0).a << endl;

  // nonexistent event
  t = ev.to(1,3);
  id = ev.id();
  pThing.getByLabel(ev,"Thing");
  cout << t << " Run " << id.run() << " event " << id.event() << " " << pThing.ref().at(0).a << endl;

  t = ev.to(10,2);
  id = ev.id();
  pThing.getByLabel(ev,"Thing");
  cout << t << " Run " << id.run() << " event " << id.event() << " " << pThing.ref().at(0).a << endl;
}

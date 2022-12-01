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

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/FWLite/interface/Event.h"
#include "FWCore/Utilities/interface/InputTag.h"

#if !defined(__CINT__) && !defined(__MAKECINT__)
#include "DataFormats/TestObjects/interface/ThingCollection.h"
#endif

void event_looping_consumes_cint() {
  TFile f("good_a.root");
  fwlite::Event e(&f);

  auto token = e.consumes<std::vector<edmtest::Thing>>(edm::InputTag("Thing"));

  int i = 0;
  int returnValue = 0;
  for (; e.isValid(); ++e, ++i) {
    edm::Handle<vector<edmtest::Thing>> pThing;
    e.getByToken(token, pThing);

    for (int i = 0; i != pThing->size(); ++i) {
      cout << pThing->at(i).a << " ";
    }
    cout << endl;
  }
  if (i == 0) {
    cout << "First loop failed!" << endl;
    returnValue = 1;
  }
  exit(returnValue);
}

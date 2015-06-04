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

#include <string>
#include <vector>
#include <iostream>

void triggerResultsByName_cint()
{
  std::vector<std::string> files;
  files.push_back(std::string("prodmerge.root"));
  fwlite::ChainEvent ev(files);

  fwlite::Handle<edm::TriggerResults> hTriggerResults;


  bool expectedValue[4] = { true, false, true, true };
  int iEvent = 0;
  for (ev.toBegin(); ! ev.atEnd(); ++ev) {

    bool accept = false;
    edm::TriggerResultsByName resultsByName = ev.triggerResultsByName("TEST");
    if (resultsByName.isValid()) {
      std::cout << "From TriggerResultsByName, accept = "
                << resultsByName.accept("p") << "\n";
      accept = resultsByName.accept("p");
    }
    if (iEvent < 4 && expectedValue[iEvent] != accept) {
      std::cerr << "triggerResultsByName_cint.C, trigger results do not match expected values" << std::endl;
      abort();
    }

    // Try again, but this time test what happens when the process does not
    // exist, the object should not be valid
    resultsByName = ev.triggerResultsByName("DOESNOTEXIST");
    if (resultsByName.isValid()) {
      std::cout << "From TriggerResultsByName, accept = "
                << resultsByName.accept("p") << "\n";
    }
    ++iEvent;
  }
}

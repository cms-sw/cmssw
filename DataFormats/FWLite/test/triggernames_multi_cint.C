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

#include <string>
#include <vector>
#include <iostream>

void triggernames_multi_cint()
{
  std::vector<std::string> files1;
  files1.push_back(std::string("prodmerge.root"));

  std::vector<std::string> files2;
  files2.push_back(std::string("prod1.root"));
  files2.push_back(std::string("prod2.root"));

  fwlite::MultiChainEvent ev(files1, files2);
  fwlite::Handle<edm::TriggerResults> hTriggerResults;
  
  int iEvent = 0;
  for (ev.toBegin(); ! ev.atEnd(); ++ev) {
    ++iEvent;
    hTriggerResults.getByLabel(ev,"TriggerResults","","TEST");
    edm::TriggerNames const&  triggerNames = ev.triggerNames(*hTriggerResults);

    std::vector<std::string> const& names = triggerNames.triggerNames();
    for (unsigned i = 0; i < triggerNames.size(); ++i) {
      std::cout << names[i] << "  " << triggerNames.triggerName(i) << std::endl;
    }
    std::cout << "size = " << triggerNames.size() << std::endl;
    std::cout << "index for p = " << triggerNames.triggerIndex("p") << std::endl;
    std::cout << "index for p1 = " << triggerNames.triggerIndex("p1") << std::endl;
    std::cout << "index for p2 = " << triggerNames.triggerIndex("p2") << std::endl;

    if (iEvent == 1) {
      if (triggerNames.size() != 3U ||
          names[0] != "p" ||
          names[1] != "p1" ||
          names[2] != "p2" ||
          triggerNames.triggerName(0) != "p" ||
          triggerNames.triggerName(1) != "p1" ||
          triggerNames.triggerName(2) != "p2" ||
          triggerNames.triggerIndex("p") != 0 ||
          triggerNames.triggerIndex("p1") != 1 ||
          triggerNames.triggerIndex("p2") != 2) {
	std::cout << "Trigger names do match expected values\n";
	std::cout << "In script triggernames_multi_cint.C\n";
        exit(1);
      }
    }
  }
}

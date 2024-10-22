#include "TFile.h"
#include "TH2S.h"
#include <iostream>
#include <string>

static constexpr int expected_entries = 20;
static constexpr int expected_entries_firstrun = 10;

void checkDQMHarvesting(TString filename) {
  TFile *f = TFile::Open(filename);

  if (!f) {
    std::cout << "checkMultiRunHarvesting: ERROR. Could not find the file: " << filename << std::endl;
    exit(EXIT_FAILURE);
  } else {
    std::string searchstring = "DQMData/Run 999999/AlCaReco/Run summary/SiStripGains/EventStats";
    TH2I *h2 = (TH2I *)f->Get(searchstring.c_str());
    if (!h2) {
      std::cout << "checkMultiRunHarvesting: ERROR. Could not find the histogram " << searchstring << std::endl;
      exit(EXIT_FAILURE);
    } else {
      int entries = h2->GetBinContent(1, 1);  // this logs the number of events
      if (entries != expected_entries) {
        std::cout << "checkMultiRunHarvesting: ERROR. This (" << entries
                  << ") is not the expect amount of events: " << expected_entries << std::endl;
        if (entries == expected_entries_firstrun) {
          std::cout << "checkMultiRunHarvesting: looks like only the first run was harvested!! " << std::endl;
        }
        exit(EXIT_FAILURE);
      } else {
        std::cout << "checkMultiRunHarvesting: this (" << entries << ") corresponds to the expected number of events ("
                  << expected_entries << ")" << std::endl;
      }
    }
  }
}

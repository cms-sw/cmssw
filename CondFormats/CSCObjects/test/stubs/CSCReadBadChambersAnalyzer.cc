#include <stdexcept>
#include <string>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <bitset>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/CSCObjects/interface/CSCBadChambers.h"
#include "CondFormats/DataRecord/interface/CSCBadChambersRcd.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/CSCIndexer.h"

namespace edmtest {
  class CSCReadBadChambersAnalyzer : public edm::EDAnalyzer {
  public:
    explicit CSCReadBadChambersAnalyzer(edm::ParameterSet const& ps)
        : outputToFile_(ps.getParameter<bool>("outputToFile")),
          readBadChambers_(ps.getParameter<bool>("readBadChambers")),
          me42installed_(ps.getParameter<bool>("me42installed")) {}

    explicit CSCReadBadChambersAnalyzer(int i) {}

    ~CSCReadBadChambersAnalyzer() override {}

    void analyze(const edm::Event& e, const edm::EventSetup& c) override;

    /// did we request reading bad channel info from db?
    bool readBadChambers() const { return readBadChambers_; }

  private:
    bool outputToFile_;
    bool readBadChambers_;  // flag whether or not to even attempt reading bad channel info from db
    bool me42installed_;    // flag whether ME42 chambers are installed in the geometry
    const CSCBadChambers* theBadChambers;
  };

  void CSCReadBadChambersAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& context) {
    using namespace edm::eventsetup;

    int counter = 0;
    std::cout << " RUN# " << e.id().run() << std::endl;
    std::cout << " EVENT# " << e.id().event() << std::endl;
    edm::ESHandle<CSCBadChambers> pBad;
    context.get<CSCBadChambersRcd>().get(pBad);

    theBadChambers = pBad.product();

    CSCIndexer indexer;  // just to build a CSCDetId from chamber index

    std::cout << "Bad Chambers:" << std::endl;

    int nbad = theBadChambers->numberOfChambers();
    std::cout << "No. in list = " << nbad << std::endl;

    // Iterate over all chambers via their linear index

    int countbad = 0;
    int countgood = 0;

    // One more than total number of chambers
    // Last chamber is already in ME4 but could be 41 or 42
    int lastRing = 1;
    if (me42installed_)
      lastRing = 2;
    int totalc = indexer.startChamberIndexInEndcap(2, 4, lastRing) + indexer.chambersInRingOfStation(4, lastRing);

    for (int indexc = 1; indexc != totalc; ++indexc) {
      counter++;

      CSCDetId id = indexer.detIdFromChamberIndex(indexc);
      bool bbad = theBadChambers->isInBadChamber(id);
      std::string bbads = "LIVE";
      if (bbad) {
        bbads = "DEAD";
        ++countbad;
      } else {
        ++countgood;
      }
      std::cout << counter << "  " << indexc << " " << id << " In bad list? " << bbads << std::endl;
    }

    std::cout << "Total number of chambers      = " << counter << std::endl;
    std::cout << "Total number of good chambers = " << countgood << std::endl;
    std::cout << "Total number of bad chambers  = " << countbad << std::endl;

    if (outputToFile_) {
      std::ofstream BadChamberFile("dbBadChamber.dat", std::ios::app);
      std::vector<int> badChambers = theBadChambers->container();
      counter = 0;
      std::vector<int>::const_iterator itcham;

      for (itcham = badChambers.begin(); itcham != badChambers.end(); ++itcham) {
        counter++;
        BadChamberFile << counter << "  " << *itcham << std::endl;
      }
    }
  }

  DEFINE_FWK_MODULE(CSCReadBadChambersAnalyzer);
}  // namespace edmtest

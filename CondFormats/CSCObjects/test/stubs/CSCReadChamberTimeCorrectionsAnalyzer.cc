/*----------------------------------------------------------------------

Toy EDProducers and EDProducts for testing purposes only.

----------------------------------------------------------------------*/

#include <stdexcept>
#include <string>
#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/CSCObjects/interface/CSCChamberTimeCorrections.h"
#include "CondFormats/DataRecord/interface/CSCChamberTimeCorrectionsRcd.h"

using namespace std;

namespace edmtest {
  class CSCReadChamberTimeCorrectionsValuesAnalyzer : public edm::EDAnalyzer {
  public:
    explicit CSCReadChamberTimeCorrectionsValuesAnalyzer(edm::ParameterSet const& p) {}
    explicit CSCReadChamberTimeCorrectionsValuesAnalyzer(int i) {}
    ~CSCReadChamberTimeCorrectionsValuesAnalyzer() override {}
    void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  private:
  };

  void CSCReadChamberTimeCorrectionsValuesAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& context) {
    using namespace edm::eventsetup;
    // Context is not used.
    std::cout << " I AM IN RUN NUMBER " << e.id().run() << std::endl;
    std::cout << " ---EVENT NUMBER " << e.id().event() << std::endl;
    edm::ESHandle<CSCChamberTimeCorrections> pChamberTimeCorrections;
    context.get<CSCChamberTimeCorrectionsRcd>().get(pChamberTimeCorrections);

    const CSCChamberTimeCorrections* myChamberTimeCorrections = pChamberTimeCorrections.product();

    //    std::map<int,CSCMapItem::MapItem>::const_iterator it;
    std::vector<CSCChamberTimeCorrections::ChamberTimeCorrections>::const_iterator it;

    int count = 0;
    for (it = myChamberTimeCorrections->chamberCorrections.begin();
         it != myChamberTimeCorrections->chamberCorrections.end();
         ++it) {
      count = count + 1;
      //      std::cout<<"Key: ddu_crate*10+ddu_input "<<it->first<<std::endl;

      std::cout << count << ") ";
      //      std::cout<<it->chamber_label<<"  ";
      std::cout << it->cfeb_length * 1. / myChamberTimeCorrections->factor_precision << "  ";
      std::cout << it->cfeb_rev << "  ";
      std::cout << it->alct_length * 1. / myChamberTimeCorrections->factor_precision << "  ";
      std::cout << it->cfeb_tmb_skew_delay * 1. / myChamberTimeCorrections->factor_precision << "  ";
      std::cout << it->cfeb_timing_corr * 1. / myChamberTimeCorrections->factor_precision << "  ";
      std::cout << "delay " << it->cfeb_cable_delay << std::endl;
    }
  }
  DEFINE_FWK_MODULE(CSCReadChamberTimeCorrectionsValuesAnalyzer);
}  // namespace edmtest

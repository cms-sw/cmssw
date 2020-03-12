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
//#include "CondFormats/CSCObjects/interface/CSCMapItem.h"
//#include "OnlineDB/CSCCondDB/interface/CSCCableRead.h"

using namespace std;

namespace edmtest {
  class CSCReadChamberTimeCorrAnalyzer : public edm::EDAnalyzer {
  public:
    explicit CSCReadChamberTimeCorrAnalyzer(edm::ParameterSet const& p) {}
    explicit CSCReadChamberTimeCorrAnalyzer(int i) {}
    virtual ~CSCReadChamberTimeCorrAnalyzer() {}
    virtual void analyze(const edm::Event& e, const edm::EventSetup& c);

  private:
  };

  void CSCReadChamberTimeCorrAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& context) {
    using namespace edm::eventsetup;
    // Context is not used.
    std::cout << " I AM IN RUN NUMBER " << e.id().run() << std::endl;
    std::cout << " ---EVENT NUMBER " << e.id().event() << std::endl;
    edm::ESHandle<CSCChamberTimeCorrections> pCorr;
    context.get<CSCChamberTimeCorrectionsRcd>().get(pCorr);

    const CSCChamberTimeCorrections* myCorr = pCorr.product();

    //    std::map<int,CSCMapItem::MapItem>::const_iterator it;
    std::vector<CSCChamberTimeCorrections::ChamberTimeCorrections>::const_iterator it;

    int count = 0;
    for (it = myCorr->chamberCorrections.begin(); it != myCorr->chamberCorrections.end(); ++it) {
      count = count + 1;
      //      std::cout<<"Key: ddu_crate*10+ddu_input "<<it->first<<std::endl;

      std::cout << count << ") ";
      //      std::cout<<it->chamber_label<<"  ";
      std::cout << it->cfeb_length << "  ";
      std::cout << it->cfeb_rev << "  ";
      std::cout << it->alct_length << "  ";
      std::cout << it->alct_rev << "  ";
      std::cout << it->cfeb_tmb_skew_delay << "  ";
      std::cout << it->anode_bx_offset << "  ";
      std::cout << it->cfeb_timing_corr << "  " << std::endl;
    }
  }
  DEFINE_FWK_MODULE(CSCReadChamberTimeCorrAnalyzer);
}  // namespace edmtest

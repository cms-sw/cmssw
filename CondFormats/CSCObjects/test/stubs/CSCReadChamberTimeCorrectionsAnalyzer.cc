/*----------------------------------------------------------------------

Toy EDProducers and EDProducts for testing purposes only.

----------------------------------------------------------------------*/

#include <stdexcept>
#include <string>
#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/CSCObjects/interface/CSCChamberTimeCorrections.h"
#include "CondFormats/DataRecord/interface/CSCChamberTimeCorrectionsRcd.h"

using namespace std;

namespace edmtest {
  class CSCReadChamberTimeCorrectionsValuesAnalyzer : public edm::global::EDAnalyzer<> {
  public:
    explicit CSCReadChamberTimeCorrectionsValuesAnalyzer(edm::ParameterSet const& p) : token_{esConsumes()} {}
    explicit CSCReadChamberTimeCorrectionsValuesAnalyzer(int i) {}
    ~CSCReadChamberTimeCorrectionsValuesAnalyzer() override {}
    void analyze(edm::StreamID, const edm::Event& e, const edm::EventSetup& c) const override;

  private:
    edm::ESGetToken<CSCChamberTimeCorrections, CSCChamberTimeCorrectionsRcd> token_;
  };

  void CSCReadChamberTimeCorrectionsValuesAnalyzer::analyze(edm::StreamID,
                                                            const edm::Event& e,
                                                            const edm::EventSetup& context) const {
    using namespace edm::eventsetup;

    edm::LogSystem log("CSCChamberTimeCorrections");
    log << " I AM IN RUN NUMBER " << e.id().run() << std::endl;
    log << " ---EVENT NUMBER " << e.id().event() << std::endl;

    const CSCChamberTimeCorrections* myChamberTimeCorrections = &context.getData(token_);

    //    std::map<int,CSCMapItem::MapItem>::const_iterator it;
    std::vector<CSCChamberTimeCorrections::ChamberTimeCorrections>::const_iterator it;

    int count = 0;
    for (it = myChamberTimeCorrections->chamberCorrections.begin();
         it != myChamberTimeCorrections->chamberCorrections.end();
         ++it) {
      count = count + 1;
      //      log<<"Key: ddu_crate*10+ddu_input "<<it->first<<std::endl;

      log << count << ") ";
      //      log<<it->chamber_label<<"  ";
      log << it->cfeb_length * 1. / myChamberTimeCorrections->factor_precision << "  ";
      log << it->cfeb_rev << "  ";
      log << it->alct_length * 1. / myChamberTimeCorrections->factor_precision << "  ";
      log << it->cfeb_tmb_skew_delay * 1. / myChamberTimeCorrections->factor_precision << "  ";
      log << it->cfeb_timing_corr * 1. / myChamberTimeCorrections->factor_precision << "  ";
      log << "delay " << it->cfeb_cable_delay << std::endl;
    }
  }
  DEFINE_FWK_MODULE(CSCReadChamberTimeCorrectionsValuesAnalyzer);
}  // namespace edmtest

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
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/CSCObjects/interface/CSCChamberTimeCorrections.h"
#include "CondFormats/DataRecord/interface/CSCChamberTimeCorrectionsRcd.h"
//#include "CondFormats/CSCObjects/interface/CSCMapItem.h"
//#include "OnlineDB/CSCCondDB/interface/CSCCableRead.h"

using namespace std;

namespace edmtest {
  class CSCReadChamberTimeCorrAnalyzer : public edm::global::EDAnalyzer<> {
  public:
    explicit CSCReadChamberTimeCorrAnalyzer(edm::ParameterSet const& p) : corrToken_{esConsumes()} {}
    ~CSCReadChamberTimeCorrAnalyzer() override {}
    void analyze(edm::StreamID, const edm::Event& e, const edm::EventSetup& c) const override;

  private:
    edm::ESGetToken<CSCChamberTimeCorrections, CSCChamberTimeCorrectionsRcd> corrToken_;
  };

  void CSCReadChamberTimeCorrAnalyzer::analyze(edm::StreamID,
                                               const edm::Event& e,
                                               const edm::EventSetup& context) const {
    using namespace edm::eventsetup;

    edm::LogSystem log("CSCCamberTimeCorrections");

    log << " I AM IN RUN NUMBER " << e.id().run() << std::endl;
    log << " ---EVENT NUMBER " << e.id().event() << std::endl;

    const CSCChamberTimeCorrections* myCorr = &context.getData(corrToken_);

    //    std::map<int,CSCMapItem::MapItem>::const_iterator it;
    std::vector<CSCChamberTimeCorrections::ChamberTimeCorrections>::const_iterator it;

    int count = 0;
    for (it = myCorr->chamberCorrections.begin(); it != myCorr->chamberCorrections.end(); ++it) {
      count = count + 1;
      //      log<<"Key: ddu_crate*10+ddu_input "<<it->first<<std::endl;

      log << count << ") ";
      //      log<<it->chamber_label<<"  ";
      log << it->cfeb_length << "  ";
      log << it->cfeb_rev << "  ";
      log << it->alct_length << "  ";
      log << it->alct_rev << "  ";
      log << it->cfeb_tmb_skew_delay << "  ";
      log << it->anode_bx_offset << "  ";
      log << it->cfeb_timing_corr << "  " << std::endl;
    }
  }
  DEFINE_FWK_MODULE(CSCReadChamberTimeCorrAnalyzer);
}  // namespace edmtest

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "OnlineDB/CSCCondDB/interface/CSCCableRead.h"

class CSCChamberTimeCorrectionsReadTest : public edm::EDAnalyzer {
public:
  explicit CSCChamberTimeCorrectionsReadTest(const edm::ParameterSet&);
  ~CSCChamberTimeCorrectionsReadTest() override;

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(CSCChamberTimeCorrectionsReadTest);

CSCChamberTimeCorrectionsReadTest::CSCChamberTimeCorrectionsReadTest(const edm::ParameterSet &) {}
CSCChamberTimeCorrectionsReadTest::~CSCChamberTimeCorrectionsReadTest() {}

void CSCChamberTimeCorrectionsReadTest::analyze(const edm::Event &, const edm::EventSetup &) {
  csccableread *cable = new csccableread();
  std::cout << " Connected cscr for cables ... " << std::endl;

  // Get information by chamber index.
  int chamberindex = 2;
  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << "Method cable_read, input: chamber index  " << chamberindex << std::endl;
  std::cout << std::endl;
  std::string chamber_label, cfeb_rev, alct_rev;
  float cfeb_length, alct_length, cfeb_tmb_skew_delay, cfeb_timing_corr;

  for (int i = 1; i <= 540; i++) {
    cable->cable_read(
        i, &chamber_label, &cfeb_length, &cfeb_rev, &alct_length, &alct_rev, &cfeb_tmb_skew_delay, &cfeb_timing_corr);

    std::cout << "chamber_label  "
              << "  " << chamber_label << " ";  //<<std::endl;
    std::cout << "cfeb_length  "
              << "  " << cfeb_length << " ";  //<<std::endl;
    std::cout << "cfeb_rev  "
              << "  " << cfeb_rev << " ";  //<<std::endl;
    std::cout << "alct_length  "
              << "  " << alct_length << " ";  //<<std::endl;
    std::cout << "alct_rev  "
              << "  " << alct_rev << " ";  //<<std::endl;
    std::cout << "cfeb_tmb_skew_delay  "
              << "  " << cfeb_tmb_skew_delay << " ";  //<<std::endl;
    std::cout << "cfeb_timing_corr  "
              << "  " << cfeb_timing_corr << std::endl;
  }
}
void CSCChamberTimeCorrectionsReadTest::beginJob() {
  std::cout << "Here is the start" << std::endl;
  std::cout << "-----------------" << std::endl;
}
void CSCChamberTimeCorrectionsReadTest::endJob() {
  std::cout << "---------------" << std::endl;
  std::cout << "Here is the end" << std::endl;
}

// system include files
#include <memory>
#include <iostream>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CondCore/CondDB/interface/Time.h"

#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSRPAlignmentCorrectionsData.h"
#include "CondFormats/AlignmentRecord/interface/CTPPSRPAlignmentCorrectionsDataRcd.h"
#include "CondFormats/AlignmentRecord/interface/RPRealAlignmentRecord.h"
#include "CondFormats/AlignmentRecord/interface/RPMisalignedAlignmentRecord.h"

using namespace std;

class CTPPSRPAlignmentInfoReader : public edm::one::EDAnalyzer<> {
public:
  cond::Time_t iov_;
  std::string record_;

  explicit CTPPSRPAlignmentInfoReader(edm::ParameterSet const& iConfig)
      : iov_(iConfig.getParameter<unsigned long long>("iov")), record_(iConfig.getParameter<string>("record")) {}
  explicit CTPPSRPAlignmentInfoReader(int i) {}
  ~CTPPSRPAlignmentInfoReader() override {}
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;
  void printInfo(const CTPPSRPAlignmentCorrectionsData& alignments, const edm::Event& event);
};

void CTPPSRPAlignmentInfoReader::analyze(const edm::Event& e, const edm::EventSetup& context) {
  using namespace edm;

  //this part gets the handle of the event source and the record (i.e. the Database)
  if (e.id().run() == iov_) {
    ESHandle<CTPPSRPAlignmentCorrectionsData> alignments;
    if (strcmp(record_.c_str(), "CTPPSRPAlignmentCorrectionsDataRcd") == 0) {
      context.get<CTPPSRPAlignmentCorrectionsDataRcd>().get(alignments);
    } else if (strcmp(record_.c_str(), "RPRealAlignmentRecord") == 0) {
      context.get<RPRealAlignmentRecord>().get(alignments);
    } else {
      context.get<RPMisalignedAlignmentRecord>().get(alignments);
    }

    //std::cout
    edm::LogPrint("CTPPSRPAlignmentInfoReader")
        << "New alignments found in run=" << e.id().run() << ", event=" << e.id().event() << ":\n"
        << *alignments;
  }
}

DEFINE_FWK_MODULE(CTPPSRPAlignmentInfoReader);

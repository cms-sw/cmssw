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

#include "CondFormats/PPSObjects/interface/CTPPSRPAlignmentCorrectionsData.h"
#include "CondFormats/AlignmentRecord/interface/CTPPSRPAlignmentCorrectionsDataRcd.h"
#include "CondFormats/AlignmentRecord/interface/RPRealAlignmentRecord.h"
#include "CondFormats/AlignmentRecord/interface/RPMisalignedAlignmentRecord.h"

using namespace std;

class CTPPSRPAlignmentInfoReader : public edm::one::EDAnalyzer<> {
public:
  cond::Time_t iov_;
  std::string record_;

  edm::ESGetToken<CTPPSRPAlignmentCorrectionsData, CTPPSRPAlignmentCorrectionsDataRcd> tokenAlignmentsIdeal_;
  edm::ESGetToken<CTPPSRPAlignmentCorrectionsData, RPRealAlignmentRecord> tokenAlignmentsReal_;
  edm::ESGetToken<CTPPSRPAlignmentCorrectionsData, RPMisalignedAlignmentRecord> tokenAlignmentsMisaligned_;

  explicit CTPPSRPAlignmentInfoReader(edm::ParameterSet const& iConfig);

  explicit CTPPSRPAlignmentInfoReader(int i) {}
  ~CTPPSRPAlignmentInfoReader() override {}
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;
  void printInfo(const CTPPSRPAlignmentCorrectionsData& alignments, const edm::Event& event);
};

//----------------------------------------------------------------------------------------------------

CTPPSRPAlignmentInfoReader::CTPPSRPAlignmentInfoReader(edm::ParameterSet const& iConfig)
    : iov_(iConfig.getParameter<unsigned long long>("iov")), record_(iConfig.getParameter<string>("record")) {
  if (strcmp(record_.c_str(), "CTPPSRPAlignmentCorrectionsDataRcd") == 0) {
    tokenAlignmentsIdeal_ = esConsumes<CTPPSRPAlignmentCorrectionsData, CTPPSRPAlignmentCorrectionsDataRcd>();
  } else if (strcmp(record_.c_str(), "RPRealAlignmentRecord") == 0) {
    tokenAlignmentsReal_ = esConsumes<CTPPSRPAlignmentCorrectionsData, RPRealAlignmentRecord>();
  } else {
    tokenAlignmentsMisaligned_ = esConsumes<CTPPSRPAlignmentCorrectionsData, RPMisalignedAlignmentRecord>();
  }
}

//----------------------------------------------------------------------------------------------------

void CTPPSRPAlignmentInfoReader::analyze(const edm::Event& e, const edm::EventSetup& context) {
  using namespace edm;

  //this part gets the handle of the event source and the record (i.e. the Database)
  if (e.id().run() == iov_) {
    ESHandle<CTPPSRPAlignmentCorrectionsData> hAlignments;

    if (strcmp(record_.c_str(), "CTPPSRPAlignmentCorrectionsDataRcd") == 0) {
      hAlignments = context.getHandle(tokenAlignmentsIdeal_);
    } else if (strcmp(record_.c_str(), "RPRealAlignmentRecord") == 0) {
      hAlignments = context.getHandle(tokenAlignmentsReal_);
    } else {
      hAlignments = context.getHandle(tokenAlignmentsMisaligned_);
    }

    //std::cout
    edm::LogPrint("CTPPSRPAlignmentInfoReader")
        << "New alignments found in run=" << e.id().run() << ", event=" << e.id().event() << ":\n"
        << *hAlignments;
  }
}

DEFINE_FWK_MODULE(CTPPSRPAlignmentInfoReader);

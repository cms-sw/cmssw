/****************************************************************************
 *
 * Authors:
 *  Jan Kaspar
 * Adapted by:
 *  Helena Malbouisson
 *  Clemencia Mora Herrera  
 ****************************************************************************/

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "CondCore/CondDB/interface/Time.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "CondFormats/AlignmentRecord/interface/CTPPSRPAlignmentCorrectionsDataRcd.h"
#include "CondFormats/AlignmentRecord/interface/RPRealAlignmentRecord.h"
#include "CondFormats/AlignmentRecord/interface/RPMisalignedAlignmentRecord.h"

#include "CondFormats/PPSObjects/interface/CTPPSRPAlignmentCorrectionsData.h"
#include <string>

//----------------------------------------------------------------------------------------------------

/**
 * \brief Class to print out information on current geometry.
 **/
class CTPPSRPAlignmentInfoAnalyzer : public edm::one::EDAnalyzer<> {
public:
  CTPPSRPAlignmentInfoAnalyzer(const edm::ParameterSet& ps);
  ~CTPPSRPAlignmentInfoAnalyzer() override {}

private:
  void analyze(const edm::Event& e, const edm::EventSetup& es) override;

  void printInfo(const CTPPSRPAlignmentCorrectionsData& alignments, const edm::Event& event) const;

  edm::ESGetToken<CTPPSRPAlignmentCorrectionsData, CTPPSRPAlignmentCorrectionsDataRcd> tokenAlignmentIdeal_;
  edm::ESGetToken<CTPPSRPAlignmentCorrectionsData, RPRealAlignmentRecord> tokenAlignmentReal_;
  edm::ESGetToken<CTPPSRPAlignmentCorrectionsData, RPMisalignedAlignmentRecord> tokenAlignmentMisaligned_;

  std::string record_;
  cond::Time_t iov_;
};

//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

using namespace std;
using namespace edm;

//----------------------------------------------------------------------------------------------------

CTPPSRPAlignmentInfoAnalyzer::CTPPSRPAlignmentInfoAnalyzer(const edm::ParameterSet& iConfig)
    : record_(iConfig.getParameter<string>("record")), iov_(iConfig.getParameter<unsigned long long>("iov")) {
  if (strcmp(record_.c_str(), "CTPPSRPAlignmentCorrectionsDataRcd") == 0) {
    tokenAlignmentIdeal_ = esConsumes<CTPPSRPAlignmentCorrectionsData, CTPPSRPAlignmentCorrectionsDataRcd>();
  } else if (strcmp(record_.c_str(), "RPRealAlignmentRecord") == 0) {
    tokenAlignmentReal_ = esConsumes<CTPPSRPAlignmentCorrectionsData, RPRealAlignmentRecord>();
  } else {
    tokenAlignmentMisaligned_ = esConsumes<CTPPSRPAlignmentCorrectionsData, RPMisalignedAlignmentRecord>();
  }
}

//----------------------------------------------------------------------------------------------------

void CTPPSRPAlignmentInfoAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const auto alignments = [&r = record_,
                           &eS = iSetup,
                           &tAI = tokenAlignmentIdeal_,
                           &tAR = tokenAlignmentReal_,
                           &tAM = tokenAlignmentMisaligned_]() -> const CTPPSRPAlignmentCorrectionsData {
    if (r == "CTPPSRPAlignmentCorrectionsDataRcd") {
      return eS.getData(tAI);
    } else if (r == "RPRealAlignmentRecord") {
      return eS.getData(tAR);
    } else {
      return eS.getData(tAM);
    }
  }();

  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if (!poolDbService.isAvailable()) {
    edm::LogError("CTPPSAlignmentInfoAnalyzer") << " DbService not available ";
  } else {
    poolDbService->writeOneIOV(alignments, iov_, record_);
  }
}

//----------------------------------------------------------------------------------------------------

void CTPPSRPAlignmentInfoAnalyzer::printInfo(const CTPPSRPAlignmentCorrectionsData& alignments,
                                             const edm::Event& event) const {
  time_t unixTime = event.time().unixTime();
  char timeStr[50];
  strftime(timeStr, 50, "%F %T", localtime(&unixTime));

  edm::LogInfo("CTPPSRPAlignmentInfoAnalyzer")
      << "New  alignments found in run=" << event.id().run() << ", event=" << event.id().event()
      << ", UNIX timestamp=" << unixTime << " (" << timeStr << "):\n"
      << alignments;
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(CTPPSRPAlignmentInfoAnalyzer);

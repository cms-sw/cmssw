#include "../interface/CertificationClient.h"

#include "DQM/EcalBarrelMonitorTasks/interface/IntegrityTask.h"
#include "DQM/EcalBarrelMonitorTasks/interface/RawDataTask.h"
#include "DQM/EcalBarrelMonitorTasks/interface/TowerStatusTask.h"
#include "DQM/EcalBarrelMonitorClient/interface/SummaryClient.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

namespace ecaldqm {

  CertificationClient::CertificationClient(const edm::ParameterSet& _params, const edm::ParameterSet& _paths) :
    DQWorkerClient(_params, _paths, "CertificationClient")
  {
    edm::ParameterSet const& sources(_params.getUntrackedParameterSet("sources"));
    source_(sIntegrity, "IntegrityTask", IntegrityTask::kByLumi, sources);
    source_(sFEStatus, "RawDataTask", RawDataTask::kFEStatus, sources);
    source_(sDesync, "RawDataTask", RawDataTask::kDesyncByLumi, sources);
    source_(sDAQ, "TowerStatusTask", TowerStatusTask::kDAQSummaryMap, sources);
    source_(sDCS, "TowerStatusTask", TowerStatusTask::kDCSSummaryMap, sources);
  }

  void
  CertificationClient::bookMEs()
  {
    unsigned iMEs[] = {kCertificationMap, kCertificationContents, kCertification};
    for(unsigned iME(0); iME < sizeof(iMEs) / sizeof(unsigned); iME++)
      MEs_[iMEs[iME]]->book();

    MEs_[kReportSummaryMap]->retrieve();
    MEs_[kReportSummaryContents]->retrieve();
    MEs_[kReportSummary]->retrieve();
  }

  void
  CertificationClient::beginRun(const edm::Run &, const edm::EventSetup &)
  {
    MEs_[kCertificationMap]->resetAll(-1.);
    MEs_[kCertificationMap]->reset(1.);
    MEs_[kCertificationContents]->reset(1.);
    MEs_[kCertification]->reset(1.);
  }

  void
  CertificationClient::producePlots()
  {
    MEs_[kCertificationMap]->reset(1.);
    for(unsigned dccid(1); dccid <= 54; dccid)
      MEs_[kCertificationContents]->fill(dccid, 1.);

    MEs_[kCertification]->fill(1.);
  }

  /*static*/
  void
  CertificationClient::setMEData(std::vector<MEData>& _data)
  {
    _data[kCertificationMap] = MEData("CertificationMap", BinService::kEcal, BinService::kSuperCrystal, MonitorElement::DQM_KIND_TH2F);
    _data[kCertificationContents] = MEData("CertificationContents", BinService::kSM, BinService::kReport, MonitorElement::DQM_KIND_REAL);
    _data[kCertification] = MEData("Certification", BinService::kEcal, BinService::kReport, MonitorElement::DQM_KIND_REAL);
    _data[kReportSummaryMap] = MEData("ReportSummaryMap", BinService::kEcal, BinService::kSuperCrystal, MonitorElement::DQM_KIND_TH2F);
    _data[kReportSummaryContents] = MEData("ReportSummaryContents", BinService::kSM, BinService::kReport, MonitorElement::DQM_KIND_REAL);
    _data[kReportSummary] = MEData("ReportSummary", BinService::kEcal, BinService::kReport, MonitorElement::DQM_KIND_REAL);
  }

  DEFINE_ECALDQM_WORKER(CertificationClient);
}


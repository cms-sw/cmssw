#include "../interface/CertificationClient.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

namespace ecaldqm {

  CertificationClient::CertificationClient(const edm::ParameterSet& _params) :
    DQWorkerClient(_params, "CertificationClient")
  {
  }

  void
  CertificationClient::bookMEs()
  {
    DQWorker::bookMEs();

    MEs_[kCertificationMap]->resetAll(-1.);
  }

  void
  CertificationClient::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
  {
    MEs_[kCertificationMap]->reset(1.);
    MEs_[kCertificationContents]->reset(1.);
    MEs_[kCertification]->reset(1.);
  }

  void
  CertificationClient::producePlots()
  {

  }

  /*static*/
  void
  CertificationClient::setMEData(std::vector<MEData>& _data)
  {
    _data[kCertificationMap] = MEData("CertificationMap", BinService::kEcal, BinService::kSuperCrystal, MonitorElement::DQM_KIND_TH2F);
    _data[kCertificationContents] = MEData("CertificationContents", BinService::kSM, BinService::kReport, MonitorElement::DQM_KIND_REAL);
    _data[kCertification] = MEData("Certification", BinService::kEcal, BinService::kReport, MonitorElement::DQM_KIND_REAL);

    _data[sDAQ + nTargets] = MEData("DAQ");
    _data[sDCS + nTargets] = MEData("DCS");
    _data[sReport + nTargets] = MEData("Report");
  }

  DEFINE_ECALDQM_WORKER(CertificationClient);
}


#include "../interface/CertificationClient.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

namespace ecaldqm {

  CertificationClient::CertificationClient(edm::ParameterSet const& _workerParams, edm::ParameterSet const& _commonParams) :
    DQWorkerClient(_workerParams, _commonParams, "CertificationClient")
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
  CertificationClient::setMEOrdering(std::map<std::string, unsigned>& _nameToIndex)
  {
    _nameToIndex["CertificationMap"] = kCertificationMap;
    _nameToIndex["CertificationContents"] = kCertificationContents;
    _nameToIndex["Certification"] = kCertification;

    _nameToIndex["DAQ"] = kDAQ;
    _nameToIndex["DCS"] = kDCS;
    _nameToIndex["Report"] = kReport;
  }

  DEFINE_ECALDQM_WORKER(CertificationClient);
}



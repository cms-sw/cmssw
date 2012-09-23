#include "../interface/CertificationClient.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

namespace ecaldqm {

  CertificationClient::CertificationClient(edm::ParameterSet const& _workerParams, edm::ParameterSet const& _commonParams) :
    DQWorkerClient(_workerParams, _commonParams, "CertificationClient")
  {
    qualitySummaries_.insert(kCertificationMap);
    qualitySummaries_.insert(kCertificationContents);
    qualitySummaries_.insert(kCertification);
  }

  void
  CertificationClient::producePlots()
  {
    double meanValue(0.);
    for(unsigned iDCC(0); iDCC < BinService::nDCC; ++iDCC){
      double certValue(sources_[kDAQ]->getBinContent(iDCC + 1) *
                       sources_[kDCS]->getBinContent(iDCC + 1) *
                       sources_[kDQM]->getBinContent(iDCC + 1));

      MEs_[kCertificationContents]->fill(iDCC + 1, certValue);
      MEs_[kCertificationMap]->setBinContent(iDCC + 1, certValue);

      meanValue += certValue * nCrystals(iDCC + 1);
    }

    MEs_[kCertification]->fill(meanValue / EBDetId::kSizeForDenseIndexing + EEDetId::kSizeForDenseIndexing);
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
    _nameToIndex["DQM"] = kDQM;
  }

  DEFINE_ECALDQM_WORKER(CertificationClient);
}



#include "../interface/CertificationClient.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

namespace ecaldqm {

  CertificationClient::CertificationClient(edm::ParameterSet const& _workerParams, edm::ParameterSet const& _commonParams) :
    DQWorkerClient(_workerParams, _commonParams, "CertificationClient")
  {
    qualitySummaries_.insert("CertificationMap");
    qualitySummaries_.insert("CertificationContents");
    qualitySummaries_.insert("Certification");
  }

  void
  CertificationClient::producePlots()
  {
    MESet* meCertificationContents(MEs_["CertificationContents"]);
    MESet* meCertificationMap(MEs_["CertificationMap"]);
    MESet* meCertification(MEs_["Certification"]);

    MESet const* sDAQ(sources_["DAQ"]);
    MESet const* sDCS(sources_["DCS"]);
    MESet const* sDQM(sources_["DQM"]);

    double meanValue(0.);
    for(unsigned iDCC(0); iDCC < BinService::nDCC; ++iDCC){
      double certValue(sDAQ->getBinContent(iDCC + 1) *
                       sDCS->getBinContent(iDCC + 1) *
                       sDQM->getBinContent(iDCC + 1));

      meCertificationContents->fill(iDCC + 1, certValue);
      meCertificationMap->setBinContent(iDCC + 1, certValue);

      meanValue += certValue * nCrystals(iDCC + 1);
    }

    meCertification->fill(meanValue / EBDetId::kSizeForDenseIndexing + EEDetId::kSizeForDenseIndexing);
  }

  DEFINE_ECALDQM_WORKER(CertificationClient);
}



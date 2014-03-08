#include "../interface/CertificationClient.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

namespace ecaldqm
{

  CertificationClient::CertificationClient() :
    DQWorkerClient()
  {
    qualitySummaries_.insert("CertificationMap");
    qualitySummaries_.insert("CertificationContents");
    qualitySummaries_.insert("Certification");
  }

  void
  CertificationClient::producePlots(ProcessType)
  {
    MESet& meCertificationContents(MEs_.at("CertificationContents"));
    MESet& meCertificationMap(MEs_.at("CertificationMap"));
    MESet& meCertification(MEs_.at("Certification"));

    MESet const& sDAQ(sources_.at("DAQ"));
    MESet const& sDCS(sources_.at("DCS"));
    MESet const& sDQM(sources_.at("DQM"));

    double meanValue(0.);
    for(int iDCC(0); iDCC < nDCC; ++iDCC){
      double certValue(sDAQ.getBinContent(iDCC + 1) *
                       sDCS.getBinContent(iDCC + 1) *
                       sDQM.getBinContent(iDCC + 1));

      meCertificationContents.fill(iDCC + 1, certValue);
      meCertificationMap.setBinContent(iDCC + 1, certValue);

      meanValue += certValue * nCrystals(iDCC + 1);
    }

    meCertification.fill(meanValue / nChannels);
  }

  DEFINE_ECALDQM_WORKER(CertificationClient);
}



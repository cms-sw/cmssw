#include "DQM/RPCMonitorClient/interface/RPCDataCertification.h"
#include "DQM/RPCMonitorClient/interface/RPCSummaryMapHisto.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <fmt/format.h>

RPCDataCertification::RPCDataCertification(const edm::ParameterSet& ps) {
  numberOfDisks_ = ps.getUntrackedParameter<int>("NumberOfEndcapDisks", 4);
  FEDRange_.first = ps.getUntrackedParameter<unsigned int>("MinimumRPCFEDId", 790);
  FEDRange_.second = ps.getUntrackedParameter<unsigned int>("MaximumRPCFEDId", 792);
  NumberOfFeds_ = FEDRange_.second - FEDRange_.first + 1;
  offlineDQM_ = ps.getUntrackedParameter<bool>("OfflineDQM", true);

  runInfoToken_ = esConsumes<edm::Transition::EndLuminosityBlock>();

  init_ = false;
  defaultValue_ = 1.;
}

void RPCDataCertification::beginJob() {}

void RPCDataCertification::dqmEndLuminosityBlock(DQMStore::IBooker& ibooker,
                                                 DQMStore::IGetter& igetter,
                                                 edm::LuminosityBlock const& LB,
                                                 edm::EventSetup const& setup) {
  if (!init_) {
    this->checkFED(setup);

    if (!offlineDQM_) {
      this->myBooker(ibooker);
    }
  }
}

void RPCDataCertification::dqmEndJob(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {
  if (offlineDQM_) {
    this->myBooker(ibooker);
  }
}

void RPCDataCertification::checkFED(edm::EventSetup const& setup) {
  double defaultValue = 1.;

  if (auto runInfoRec = setup.tryToGet<RunInfoRcd>()) {
    defaultValue = -1;
    //get fed summary information
    auto sumFED = runInfoRec->get(runInfoToken_);
    const std::vector<int> FedsInIds = sumFED.m_fed_in;
    unsigned int f = 0;
    bool flag = false;
    while (!flag && f < FedsInIds.size()) {
      int fedID = FedsInIds[f];
      //make sure fed id is in allowed range
      if (fedID >= FEDRange_.first && fedID <= FEDRange_.second) {
        defaultValue = 1;
        flag = true;
      }
      f++;
    }
  }

  defaultValue_ = defaultValue;

  init_ = true;
}

void RPCDataCertification::myBooker(DQMStore::IBooker& ibooker) {
  ibooker.setCurrentFolder("RPC/EventInfo");
  // global fraction
  totalCertFraction = ibooker.bookFloat("CertificationSummary");
  totalCertFraction->Fill(defaultValue_);

  CertMap_ = RPCSummaryMapHisto::book(ibooker, "CertificationSummaryMap", "RPC Certification Summary Map");

  //fill the histo with "1" --- just for the moment
  RPCSummaryMapHisto::setBinsBarrel(CertMap_, defaultValue_);
  RPCSummaryMapHisto::setBinsEndcap(CertMap_, defaultValue_);

  // book the ME
  ibooker.setCurrentFolder("RPC/EventInfo/CertificationContents");

  const int limit = std::max(2, numberOfDisks_);

  for (int i = -limit; i <= limit; ++i) {  //loop on wheels and disks
    if (i > -3 && i < nWheels_ - 2) {      //wheels
      const std::string binLabel = fmt::format("RPC_Wheel{}", i);
      certWheelFractions[i + 2] = ibooker.bookFloat(binLabel);
      certWheelFractions[i + 2]->Fill(defaultValue_);
    }

    if (i == 0 || i > numberOfDisks_ || i < (-numberOfDisks_))
      continue;

    if (i > -3 && i < nDisks_ - 2) {
      const std::string binLabel = fmt::format("RPC_Disk{}", i);
      certDiskFractions[i + 2] = ibooker.bookFloat(binLabel);
      certDiskFractions[i + 2]->Fill(defaultValue_);
    }
  }
}

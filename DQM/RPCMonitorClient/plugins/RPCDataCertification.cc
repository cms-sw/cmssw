#include "DQM/RPCMonitorClient/interface/RPCDataCertification.h"

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

  CertMap_ = ibooker.book2D("CertificationSummaryMap", "RPC Certification Summary Map", 15, -7.5, 7.5, 12, 0.5, 12.5);

  //customize the 2d histo
  for (int i = 1; i < 13; i++) {
    const std::string binLabel = fmt::format("Sec{}", i);
    CertMap_->setBinLabel(i, binLabel, 2);
  }

  for (int i = -2; i <= 2; i++) {
    const std::string binLabel = fmt::format("Wheel{}", i);
    CertMap_->setBinLabel(i + 8, binLabel, 1);
  }

  for (int i = 1; i <= numberOfDisks_; i++) {
    const std::string binLabelP = fmt::format("Disk{}", i);
    CertMap_->setBinLabel((i + 11), binLabelP, 1);
    const std::string binLabelM = fmt::format("Disk{}", -i);
    CertMap_->setBinLabel((-i + 5), binLabelM, 1);
  }
  //fill the histo with "1" --- just for the moment
  for (int i = 1; i <= 15; i++) {
    for (int j = 1; j <= 12; j++) {
      if (i == 5 || i == 11 || (j > 6 && (i < 6 || i > 10)))
        CertMap_->setBinContent(i, j, -1);  //bins that not correspond to subdetector parts
      else
        CertMap_->setBinContent(i, j, defaultValue_);
    }
  }

  if (numberOfDisks_ < 4) {
    for (int j = 1; j <= 12; j++) {
      CertMap_->setBinContent(1, j, -1);  //bins that not correspond to subdetector parts
      CertMap_->setBinContent(15, j, -1);
    }
  }
  // book the ME
  ibooker.setCurrentFolder("RPC/EventInfo/CertificationContents");

  const int limit = std::max(2, numberOfDisks_);

  for (int i = -limit; i <= limit; ++i) {  //loop on wheels and disks
    if (i > -3 && i < nWheels_ - 2) {          //wheels
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

#include "DQM/RPCMonitorClient/interface/RPCDaqInfo.h"
#include "DQM/RPCMonitorClient/interface/RPCSummaryMapHisto.h"

#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CondFormats/RunInfo/interface/RunSummary.h"

#include <fmt/format.h>

RPCDaqInfo::RPCDaqInfo(const edm::ParameterSet& ps) {
  FEDRange_.first = ps.getUntrackedParameter<unsigned int>("MinimumRPCFEDId", 790);
  FEDRange_.second = ps.getUntrackedParameter<unsigned int>("MaximumRPCFEDId", 792);

  NumberOfFeds_ = FEDRange_.second - FEDRange_.first + 1;

  numberOfDisks_ = ps.getUntrackedParameter<int>("NumberOfEndcapDisks", 4);

  runInfoToken_ = esConsumes<edm::Transition::EndLuminosityBlock>();

  init_ = false;
}

void RPCDaqInfo::beginJob() {}
void RPCDaqInfo::dqmEndLuminosityBlock(DQMStore::IBooker& ibooker,
                                       DQMStore::IGetter& igetter,
                                       edm::LuminosityBlock const& LB,
                                       edm::EventSetup const& iSetup) {
  if (!init_) {
    this->myBooker(ibooker);
  }

  if (auto runInfoRec = iSetup.tryToGet<RunInfoRcd>()) {
    //get fed summary information
    auto sumFED = runInfoRec->get(runInfoToken_);
    const std::vector<int> FedsInIds = sumFED.m_fed_in;

    int FedCount = 0;

    //loop on all active feds
    for (const int fedID : FedsInIds) {
      //make sure fed id is in allowed range
      if (fedID >= FEDRange_.first && fedID <= FEDRange_.second)
        ++FedCount;
    }

    //Fill active fed fraction ME
    if (NumberOfFeds_ > 0)
      DaqFraction_->Fill(FedCount / NumberOfFeds_);
    else
      DaqFraction_->Fill(-1);

  } else {
    DaqFraction_->Fill(-1);
    return;
  }
}

void RPCDaqInfo::dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter&) {}

void RPCDaqInfo::myBooker(DQMStore::IBooker& ibooker) {
  //fraction of alive FEDs
  ibooker.setCurrentFolder("RPC/EventInfo/DAQContents");

  const int limit = std::max(2, numberOfDisks_);

  for (int i = -limit; i <= limit; ++i) {  //loop on wheels and disks
    if (i > -3 && i < nWheels_ - 2) {      //wheels
      const std::string meName = fmt::format("RPC_Wheel{}", i);
      daqWheelFractions[i + 2] = ibooker.bookFloat(meName);
      daqWheelFractions[i + 2]->Fill(-1);
    }

    if (i == 0 || i > numberOfDisks_ || i < -numberOfDisks_)
      continue;

    if (i > -3 && i < nDisks_ - 2) {
      const std::string meName = fmt::format("RPC_Disk{}", i);
      daqDiskFractions[i + 2] = ibooker.bookFloat(meName);
      daqDiskFractions[i + 2]->Fill(-1);
    }
  }

  //daq summary for RPCs
  ibooker.setCurrentFolder("RPC/EventInfo");

  DaqFraction_ = ibooker.bookFloat("DAQSummary");
  DaqMap_ = RPCSummaryMapHisto::book(ibooker, "DAQSummaryMap", "RPC DAQ Summary Map");
  RPCSummaryMapHisto::setBinsBarrel(DaqMap_, 1);
  RPCSummaryMapHisto::setBinsEndcap(DaqMap_, 1);

  init_ = true;
}

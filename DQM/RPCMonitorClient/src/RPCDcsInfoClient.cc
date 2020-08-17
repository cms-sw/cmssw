#include "DQM/RPCMonitorClient/interface/RPCDcsInfoClient.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

RPCDcsInfoClient::RPCDcsInfoClient(const edm::ParameterSet& ps) {
  dcsinfofolder_ = ps.getUntrackedParameter<std::string>("dcsInfoFolder", "RPC/DCSInfo");
  eventinfofolder_ = ps.getUntrackedParameter<std::string>("eventInfoFolder", "RPC/EventInfo");
  dqmprovinfofolder_ = ps.getUntrackedParameter<std::string>("dqmProvInfoFolder", "Info/EventInfo");

  DCS.clear();
  DCS.resize(10);  // start with 10 LS, resize later
}

RPCDcsInfoClient::~RPCDcsInfoClient() {}

void RPCDcsInfoClient::beginJob() {}

void RPCDcsInfoClient::dqmEndJob(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {
  // book
  ibooker.cd();
  ibooker.setCurrentFolder(dcsinfofolder_);

  unsigned int nlsmax = DCS.size();
  MonitorElement* reportSummaryMap_ = igetter.get(dqmprovinfofolder_ + "/reportSummaryMap");
  MonitorElement* lumiNumber_ = igetter.get(eventinfofolder_ + "/iLumiSection");

  if (!reportSummaryMap_)
    return;

  if (TH2F* h2 = reportSummaryMap_->getTH2F()) {
    nlsmax = lumiNumber_->getIntValue();
    int hvStatus = 0;
    const char* label_name = "RPC";
    unsigned int rpc_num = 0;
    if (nlsmax > DCS.size())
      DCS.resize(nlsmax);

    for (int ybin = 0; ybin < h2->GetNbinsY(); ++ybin) {
      if (strcmp(h2->GetYaxis()->GetBinLabel(ybin + 1), label_name) == 0)
        rpc_num = ybin + 1;
    }

    for (unsigned int nlumi = 0; nlumi < nlsmax; ++nlumi) {
      int rpc_dcsbit = h2->GetBinContent(nlumi + 1, rpc_num);
      if (rpc_dcsbit != -1) {
        hvStatus = 1;  // set to 1 because HV was on (!)
      } else {
        hvStatus = 0;  // set to 0 because HV was off (!)
      }
      DCS[nlumi] = hvStatus;
    }
  }

  std::string meName = dcsinfofolder_ + "/rpcHVStatus";
  unsigned int dcssize = DCS.size();
  MonitorElement* rpcHVStatus = ibooker.book2D("rpcHVStatus", "RPC HV Status", dcssize, 1., dcssize + 1, 1, 0.5, 1.5);
  rpcHVStatus->setAxisTitle("Luminosity Section", 1);
  rpcHVStatus->setBinLabel(1, "", 2);

  int lsCounter = 0;
  // fill
  for (unsigned int i = 0; i < nlsmax; i++) {
    rpcHVStatus->setBinContent(i + 1, 1, DCS[i]);
    lsCounter += DCS[i];
  }

  meName = dcsinfofolder_ + "/rpcHV";
  MonitorElement* rpcHV = ibooker.bookInt("rpcHV");

  rpcHV->Fill(lsCounter);

  return;
}

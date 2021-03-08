#include "DQM/RPCMonitorClient/interface/RPCDcsInfoClient.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

RPCDcsInfoClient::RPCDcsInfoClient(const edm::ParameterSet& ps):
  dcsinfofolder_(ps.getUntrackedParameter<std::string>("dcsInfoFolder", "RPC/DCSInfo")),
  eventinfofolder_(ps.getUntrackedParameter<std::string>("eventInfoFolder", "RPC/EventInfo")),
  dqmprovinfofolder_(ps.getUntrackedParameter<std::string>("dqmProvInfoFolder", "Info/EventInfo"))
{
}

void RPCDcsInfoClient::dqmEndJob(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {
  // book
  ibooker.cd();
  ibooker.setCurrentFolder(dcsinfofolder_);

  MonitorElement* reportSummaryMap = igetter.get(dqmprovinfofolder_ + "/reportSummaryMap");
  MonitorElement* eventInfoLumi = igetter.get(eventinfofolder_+"/iLumiSection");

  if (!reportSummaryMap)
    return;

  TH2F* h2 = reportSummaryMap->getTH2F();
  if ( !h2 ) return;
  const int maxLS = reportSummaryMap->getNbinsX();

  int nLS = eventInfoLumi->getIntValue();
  if ( nLS <= 0 or nLS > maxLS ) {
    // If the nLS from the event info is not valid, we take the value from the
    // reportSummaryMap. The histogram is initialized with -1 value then filled
    // with non-negative value for valid LSs.
    // Note that we start from the first bin, since many runs have small nLS.
    for ( nLS=1; nLS<=maxLS; ++nLS ) {
      const double dcsBit = h2->GetBinContent(nLS, 1);
      if ( dcsBit == -1 ) break;
    }
  }
  
  MonitorElement* rpcHVStatus = ibooker.book2D("rpcHVStatus", "RPC HV Status", nLS, 1., nLS+1, 1, 0.5, 1.5);
  rpcHVStatus->setAxisTitle("Luminosity Section", 1);
  rpcHVStatus->setBinLabel(1, "", 2);

  // Find bin number of RPC from the EventInfo's reportSummaryMap
  int binRPC = 0;
  for ( int i=1, nbinsY=reportSummaryMap->getNbinsY(); i<=nbinsY; ++i ) {
    const std::string binLabel = h2->GetYaxis()->GetBinLabel(i);
    if ( binLabel == "RPC" ) {
      binRPC = i;
      break;
    }
  }
  if ( binRPC == 0 ) return;

  // Take bin contents from the reportSummaryMap and fill into the RPC DCSInfo
  int nLSRPC = 0;
  for ( int i=1; i<=nLS; ++i ) {
    const double dcsBit = h2->GetBinContent(i, binRPC);
    const int hvStatus = (dcsBit != -1) ? 1 : 0;
    if ( hvStatus != 0 ) {
      ++nLSRPC;
      rpcHVStatus->setBinContent(i, 1, hvStatus);
    }
  }

  MonitorElement* rpcHV = ibooker.bookInt("rpcHV");
  rpcHV->Fill(nLSRPC);

}

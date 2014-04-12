#include "DQM/RPCMonitorClient/interface/RPCDcsInfoClient.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

RPCDcsInfoClient::RPCDcsInfoClient( const edm::ParameterSet& ps ) {

  dbe_ = edm::Service<DQMStore>().operator->();

  dcsinfofolder_ = ps.getUntrackedParameter<std::string>("dcsInfoFolder", "RPC/DCSInfo") ;
}


RPCDcsInfoClient::~RPCDcsInfoClient() {}

void RPCDcsInfoClient::beginRun(const edm::Run& r, const edm::EventSetup& c) {
  DCS.clear();
  DCS.resize(10);  // start with 10 LS, resize later
}

void RPCDcsInfoClient::analyze(const edm::Event& e, const edm::EventSetup& c){}

void RPCDcsInfoClient::endLuminosityBlock(const edm::LuminosityBlock& l, const edm::EventSetup& c){
  if (!dbe_) return;

  unsigned int nlumi = l.id().luminosityBlock() ;

  if (nlumi+1 > DCS.size())   DCS.resize(nlumi+1);


  MonitorElement* DCSbyLS_ = dbe_->get(dcsinfofolder_ + "/DCSbyLS" ); 

  if ( !DCSbyLS_ ) return;
  
  if ( TH1F * h1 = DCSbyLS_->getTH1F()) {
    int hvStatus = 0;
   
    if ( h1->GetBinContent(1) != 0 ) {
      hvStatus = 0; // set to 0 because HV was off (!)
    } else  {
      hvStatus = 1;    // set to 1 because HV was on (!)
    }

    DCS[nlumi] = hvStatus;
  }
  
  return; 
}

void RPCDcsInfoClient::endRun(const edm::Run& r, const edm::EventSetup& c) {

  // book 
  dbe_->cd();  
  dbe_->setCurrentFolder(dcsinfofolder_ );

  unsigned int nlsmax = DCS.size();
  if (nlsmax > 900 ) nlsmax = 900;
   
  std::string meName = dcsinfofolder_ + "/rpcHVStatus";
  MonitorElement* rpcHVStatus = dbe_->get(meName);
  if (rpcHVStatus) dbe_->removeElement(rpcHVStatus->getName());

  rpcHVStatus = dbe_->book2D("rpcHVStatus","RPC HV Status", nlsmax, 1., nlsmax+1, 1, 0.5, 1.5);
  rpcHVStatus->setAxisTitle("Luminosity Section", 1);
  rpcHVStatus->setBinLabel(1,"",2);   

  int lsCounter = 0;
  // fill
  for (unsigned int i = 0 ; i < nlsmax ; i++ )  {
    rpcHVStatus->setBinContent(i+1,1,DCS[i]);
    lsCounter +=DCS[i];
  }

  meName = dcsinfofolder_ + "/rpcHV";
  MonitorElement* rpcHV = dbe_->get(meName);
  if (rpcHV) dbe_->removeElement(rpcHVStatus->getName());
  rpcHV = dbe_->bookInt("rpcHV");

  rpcHV ->Fill(lsCounter);
  
  return;
}

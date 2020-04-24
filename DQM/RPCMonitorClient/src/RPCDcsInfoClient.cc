#include "DQM/RPCMonitorClient/interface/RPCDcsInfoClient.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

RPCDcsInfoClient::RPCDcsInfoClient( const edm::ParameterSet& ps ) {

   dcsinfofolder_ = ps.getUntrackedParameter<std::string>("dcsInfoFolder", "RPC/DCSInfo") ;

  DCS.clear();
  DCS.resize(10);  // start with 10 LS, resize later

}


RPCDcsInfoClient::~RPCDcsInfoClient() {}

void RPCDcsInfoClient::beginJob() {}

void RPCDcsInfoClient:: dqmEndLuminosityBlock(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter, edm::LuminosityBlock const & l, edm::EventSetup const& c){
  
  unsigned int nlumi = l.id().luminosityBlock() ;

  if (nlumi+1 > DCS.size())   DCS.resize(nlumi+1);


  MonitorElement* DCSbyLS_ = igetter.get(dcsinfofolder_ + "/DCSbyLS" ); 

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


void RPCDcsInfoClient::dqmEndJob(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter){

  // book 
  ibooker.cd();  
  ibooker.setCurrentFolder(dcsinfofolder_ );

  unsigned int nlsmax = DCS.size();
   
  std::string meName = dcsinfofolder_ + "/rpcHVStatus";
  MonitorElement* rpcHVStatus = ibooker.book2D("rpcHVStatus","RPC HV Status", nlsmax, 1., nlsmax+1, 1, 0.5, 1.5);
  rpcHVStatus->setAxisTitle("Luminosity Section", 1);
  rpcHVStatus->setBinLabel(1,"",2);   

  int lsCounter = 0;
  // fill
  for (unsigned int i = 0 ; i < nlsmax ; i++ )  {
    rpcHVStatus->setBinContent(i+1,1,DCS[i]);
    lsCounter +=DCS[i];
  }

  meName = dcsinfofolder_ + "/rpcHV";
  MonitorElement* rpcHV = ibooker.bookInt("rpcHV");

  rpcHV ->Fill(lsCounter);
  
  return;
}

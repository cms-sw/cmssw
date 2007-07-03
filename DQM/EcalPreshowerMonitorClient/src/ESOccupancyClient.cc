#include "DQM/EcalPreshowerMonitorClient/interface/ESOccupancyClient.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/MonitorElementBaseT.h"

ESOccupancyClient::ESOccupancyClient(const ParameterSet& ps) {
  
  writeHisto_ = ps.getUntrackedParameter<bool>("writeHisto", true);
  outputFile_ = ps.getUntrackedParameter<string>("outputFile", "ESOccupancy.root");
  rootFolder_ = ps.getUntrackedParameter<string>("rootFolder", "");
  
  dbe_ = Service<DaqMonitorBEInterface>().operator->();
  dbe_->setVerbose(1);
  dbe_->showDirStructure();

}

ESOccupancyClient::~ESOccupancyClient(){
}

void ESOccupancyClient::endJob(){
  
  if (writeHisto_) dbe_->save(outputFile_);
  
}

void ESOccupancyClient::beginJob(const EventSetup& context){
}

void ESOccupancyClient::analyze(const Event& e, const EventSetup& context){
	
  for (int i=0; i<2; ++i) {    

    MonitorElement * occME = dbe_->get(getMEName(i+1));
    
    if (occME) {
      MonitorElementT<TNamed>* occ = dynamic_cast<MonitorElementT<TNamed>*>(occME);           
      TH1F * h_occupancy = dynamic_cast<TH1F*> (occ->operator->());      
      cout<<h_occupancy->GetMean()<<endl;
    }

  }

}

string ESOccupancyClient::getMEName(const int & plane) {
  
  stringstream iplane; iplane << plane;

  string histoname = rootFolder_+"/ES/ESOccupancyTask/ES Occupancy Plane "+iplane.str(); 

  return histoname;

}

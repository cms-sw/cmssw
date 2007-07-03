#include "DQM/EcalPreshowerMonitorModule/interface/ESPedestalTBTask.h"

#include <iostream>

using namespace cms;
using namespace edm;
using namespace std;

ESPedestalTBTask::ESPedestalTBTask(const ParameterSet& ps) {

  label_        = ps.getUntrackedParameter<string>("Label");
  instanceName_ = ps.getUntrackedParameter<string>("InstanceES");
  sta_          = ps.getUntrackedParameter<bool>("RunStandalone", false);

  init_ = false;

  for (int i=0; i<2; ++i) 
    for (int j=0; j<4; ++j)
      for (int k=0; k<4; ++k) 
	for (int m=0; m<32; ++m)
	  mePedestal_[i][j][k][m] = 0;  

}

ESPedestalTBTask::~ESPedestalTBTask(){
}

void ESPedestalTBTask::beginJob(const EventSetup& c) {
  
  ievt_ = 0;
  
  dbe_ = Service<DaqMonitorBEInterface>().operator->();
  
  if ( dbe_ ) {
    dbe_->setCurrentFolder("ES/ESPedestalTBTask");
    dbe_->rmdir("ES/ESPedestalTBTask");
  }
  
}

void ESPedestalTBTask::setup(void){
  
  init_ = true;
  
  Char_t hist[200];
  
  dbe_ = Service<DaqMonitorBEInterface>().operator->();
  
  if ( dbe_ ) {   
    dbe_->setCurrentFolder("ES/ESPedestalTBTask");
    for (int i=0; i<2; ++i) {       
      for (int j=30; j<34; ++j) {
	for (int k=19; k<23; ++k) {
	  for (int m=0; m<32; ++m) {
	    sprintf(hist, "ES Pedestal P %d Row %02d Col %02d Str %02d", i+1, j, k, m+1);      
	    mePedestal_[i][j-30][k-19][m] = dbe_->book1D(hist, hist, 5000, 0, 5000);
	  }
	}
      }
    }
  }

}

void ESPedestalTBTask::cleanup(void){

  if (sta_) return;

  dbe_ = Service<DaqMonitorBEInterface>().operator->();

  if ( dbe_ ) {
    dbe_->setCurrentFolder("ES/ESPedestalTBTask");

    for (int i=0; i<2; ++i) {
      for (int j=0; j<4; ++j) {
	for (int k=0; k<4; ++k) {
	  for (int m=0; m<32; ++m) {
	    if ( mePedestal_[i][j][k][m] ) dbe_->removeElement( mePedestal_[i][j][k][m]->getName() );
	    mePedestal_[i][j][k][m] = 0;
	  }
	}
      }
    }

  }

  init_ = false;

}

void ESPedestalTBTask::endJob(void) {

  LogInfo("ESPedestalTBTask") << "analyzed " << ievt_ << " events";

  if ( init_ ) this->cleanup();

}

void ESPedestalTBTask::analyze(const Event& e, const EventSetup& c){

  if ( ! init_ ) this->setup();
  ievt_++;
  
  Handle<ESDigiCollection> digis;
  try {
    e.getByLabel(label_, instanceName_, digis);
  } catch ( cms::Exception &e ) {
    LogDebug("") << "ESPedestal : Error! can't get collection !" << std::endl;
  } 

  for (ESDigiCollection::const_iterator digiItr = digis->begin(); digiItr != digis->end(); ++digiItr ) {

    ESDataFrame dataframe = (*digiItr);
    ESDetId id = dataframe.id();

    int plane = id.plane();
    int ix    = id.six();
    int iy    = id.siy();
    int strip = id.strip();

    for (int i=0; i<dataframe.size(); ++i) {
      //cout<<plane<<" "<<ix<<" "<<strip<<" "<<dataframe.sample(i).adc()<<endl;    
      mePedestal_[plane-1][ix-30][iy-19][strip-1]->Fill(dataframe.sample(i).adc());
    }
  }
  
  //sleep(2);
}

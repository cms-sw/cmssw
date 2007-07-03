#include "DQM/EcalPreshowerMonitorModule/interface/ESPedestalCTTask.h"

#include <iostream>

using namespace cms;
using namespace edm;
using namespace std;

ESPedestalCTTask::ESPedestalCTTask(const ParameterSet& ps) {

  label_        = ps.getUntrackedParameter<string>("Label");
  instanceName_ = ps.getUntrackedParameter<string>("InstanceES");
  sta_          = ps.getUntrackedParameter<bool>("RunStandalone", false);

  init_ = false;

  for (int i=0; i<2; ++i) 
    for (int j=0; j<6; ++j)
      for (int k=0; k<2; ++k)        
	for (int m=0; m<5; ++m)
	  for (int n=0; n<32; ++n)
	    mePedestal_[i][j][k][m][n] = 0;  

}

ESPedestalCTTask::~ESPedestalCTTask(){
}

void ESPedestalCTTask::beginJob(const EventSetup& c) {
  
  ievt_ = 0;
  
  dbe_ = Service<DaqMonitorBEInterface>().operator->();
  
  if ( dbe_ ) {
    dbe_->setCurrentFolder("ES/ESPedestalCTTask");
    dbe_->rmdir("ES/ESPedestalCTTask");
  }
  
}

void ESPedestalCTTask::setup(void){
  
  init_ = true;
  
  Char_t hist[200];
  
  dbe_ = Service<DaqMonitorBEInterface>().operator->();
  
  if ( dbe_ ) {   
    dbe_->setCurrentFolder("ES/ESPedestalCTTask");
    for (int i=0; i<2; ++i) {       
      for (int j=0; j<6; ++j) {
	for (int k=0; k<2; ++k) {
	  for (int m=0; m<5; ++m) {
	    for (int n=0; n<32; ++n) {
	      int zside = (i==0)?1:-1;
	      sprintf(hist, "ES Pedestal Z %d P %d Row %02d Col %02d Str %02d", zside, j+1, k+1, m+1, n+1);      
	      mePedestal_[i][j][k][m][n] = dbe_->book1D(hist, hist, 5000, -0.5, 4999.5);
	    }
	  }
	}
      }
    }
  }

}

void ESPedestalCTTask::cleanup(void){

  dbe_ = Service<DaqMonitorBEInterface>().operator->();

  if ( dbe_ ) {
    dbe_->setCurrentFolder("ES/ESPedestalCTTask");

    for (int i=0; i<2; ++i) {
      for (int j=0; j<6; ++j) {
	for (int k=0; k<2; ++k) {
	  for (int m=0; m<5; ++m) {
	    for (int n=0; n<32; ++n) {
	      if ( mePedestal_[i][j][k][m][n] ) dbe_->removeElement( mePedestal_[i][j][k][m][n]->getName() );
	      mePedestal_[i][j][k][m][n] = 0;
	    }
	  }
	}
      }
    } 

  }

  init_ = false;

}

void ESPedestalCTTask::endJob(void) {

  LogInfo("ESPedestalCTTask") << "analyzed " << ievt_ << " events";

  if (sta_) return;

  if ( init_ ) this->cleanup();

}

void ESPedestalCTTask::analyze(const Event& e, const EventSetup& c){

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

    //int plane = id.plane();
    int ix    = id.six();
    int iy    = id.siy();
    int strip = id.strip();

    int j = (ix-1)/2; 
    int i;
    if (j<=5) i = 0;
    else i = 1;
    if (j>5) j = j-6;    
    int k = (ix-1)%2;

    for (int isam=0; isam<dataframe.size(); ++isam) {
      mePedestal_[i][j][k][iy-1][strip-1]->Fill(dataframe.sample(isam).adc());
    }
    
  }

  //sleep(5);
}

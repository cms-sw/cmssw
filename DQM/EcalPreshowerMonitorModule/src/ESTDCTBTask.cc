#include "DQM/EcalPreshowerMonitorModule/interface/ESTDCTBTask.h"

#include <iostream>

using namespace cms;
using namespace edm;
using namespace std;

ESTDCTBTask::ESTDCTBTask(const ParameterSet& ps) {
  
  label_        = ps.getUntrackedParameter<string>("Label");
  instanceName_ = ps.getUntrackedParameter<string>("InstanceES");
  sta_          = ps.getUntrackedParameter<bool>("RunStandalone", false);
  
  init_ = false;
  
  for (int i=0; i<2; ++i) 
    for (int j=0; j<3; ++j)
      meADC_[i][j] = 0;  

  dbe_ = Service<DaqMonitorBEInterface>().operator->();
  
}

ESTDCTBTask::~ESTDCTBTask(){
}

void ESTDCTBTask::beginJob(const EventSetup& c) {
  
  ievt_ = 0;
  
  if ( dbe_ ) {
    dbe_->setCurrentFolder("ES/ESTDCTBTask");
    dbe_->rmdir("ES/ESTDCTBTask");
  }

}

void ESTDCTBTask::setup(void){
  
  init_ = true;
  
  Char_t hist[200];
  
  if ( dbe_ ) {   
    dbe_->setCurrentFolder("ES/ESTDCTBTask");
    
    for (int i=0; i<2; ++i) {	
      for (int j=0; j<3; ++j) { 
	sprintf(hist, "ES ADC Z 1 P %d Slot %d", i+1, j+1);
	meADC_[i][j] = dbe_->book1D(hist, hist, 5200, -200, 5000);
      }
    }
  }
  
}

void ESTDCTBTask::cleanup(void){

  if (sta_) return;

  if ( dbe_ ) {
    dbe_->setCurrentFolder("ES/ESTDCTBTask");
    
    for (int i=0; i<2; ++i) {
      for (int j=0; j<3; ++j) {
	if ( meADC_[i][j] ) dbe_->removeElement( meADC_[i][j]->getName() );
	meADC_[i][j] = 0;
      }
    }
  }
 
  init_ = false;

}

void ESTDCTBTask::endJob(void) {

  LogInfo("ESTDCTBTask") << "analyzed " << ievt_ << " events";

  if ( init_ ) this->cleanup();

}

void ESTDCTBTask::analyze(const Event& e, const EventSetup& c){

  if ( ! init_ ) this->setup();
  ievt_++;
  
  Handle<ESDigiCollection> digis;
  try {
    e.getByLabel(label_, instanceName_, digis);
  } catch ( cms::Exception &e ) {
    LogDebug("") << "ESTDCTB : Error! can't get digis collection !" << std::endl;
  } 

  // Digis
  int plane, ix, iy, strip;

  for (ESDigiCollection::const_iterator digiItr = digis->begin(); digiItr != digis->end(); ++digiItr ) {
    
    ESDataFrame dataframe = (*digiItr);
    ESDetId id = dataframe.id();
    
    plane = id.plane();
    ix    = id.six();
    iy    = id.siy();
    strip = id.strip();
    
    for (int isam=0; isam<dataframe.size(); ++isam)       
      meADC_[plane-1][isam]->Fill(dataframe.sample(isam).adc());         
    
  }

}

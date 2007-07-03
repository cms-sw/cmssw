#include "DQM/EcalPreshowerMonitorModule/interface/ESOccupancyTBTask.h"

#include <iostream>

using namespace cms;
using namespace edm;
using namespace std;

ESOccupancyTBTask::ESOccupancyTBTask(const ParameterSet& ps) {

  label_ = ps.getParameter<string>("Label");
  instanceName_ = ps.getParameter<string>("InstanceES");

  init_ = false;

  for (int i=0; i<2; ++i) 
    for (int j=0; j<4; ++j)
      for (int k=0; k<4; ++k) 
	meOccupancy_[i][j][k] = 0;  

}

ESOccupancyTBTask::~ESOccupancyTBTask(){
}

void ESOccupancyTBTask::beginJob(const EventSetup& c) {
  
  ievt_ = 0;
  
  DaqMonitorBEInterface* dbe = 0;
  dbe = Service<DaqMonitorBEInterface>().operator->();
  
  if ( dbe ) {
    dbe->setCurrentFolder("ES/ESOccupancyTBTask");
    dbe->rmdir("ES/ESOccupancyTBTask");
  }
  
}

void ESOccupancyTBTask::setup(void){
  
  init_ = true;
  
  Char_t hist[200];
  
  DaqMonitorBEInterface* dbe = 0;
  dbe = Service<DaqMonitorBEInterface>().operator->();
  
  if ( dbe ) {   
    dbe->setCurrentFolder("ES/ESOccupancyTBTask");
    for (int i=0; i<2; ++i) {       
      for (int j=30; j<34; ++j) {
	for (int k=19; k<23; ++k) {
	  sprintf(hist, "ES Occupancy P %d Row %d Col %d", i+1, j, k);      
	  meOccupancy_[i][j-30][k-19] = dbe->book1D(hist, hist, 550, 0, 550);
	}
      }
    }
  }

}

void ESOccupancyTBTask::cleanup(void){

  DaqMonitorBEInterface* dbe = 0;
  dbe = Service<DaqMonitorBEInterface>().operator->();

  if ( dbe ) {
    dbe->setCurrentFolder("ES/ESOccupancyTBTask");

    for (int i=0; i<2; ++i) {
      for (int j=0; j<4; ++j) {
	for (int k=0; k<4; ++k) {
	  if ( meOccupancy_[i][j][k] ) dbe->removeElement( meOccupancy_[i][j][k]->getName() );
	  meOccupancy_[i][j][k] = 0;
	}
      }
    }

  }

  init_ = false;

}

void ESOccupancyTBTask::endJob(void) {

  LogInfo("ESOccupancyTBTask") << "analyzed " << ievt_ << " events";

  if ( init_ ) this->cleanup();

}

void ESOccupancyTBTask::analyze(const Event& e, const EventSetup& c){

  if ( ! init_ ) this->setup();
  ievt_++;
  
  Handle<ESDigiCollection> digis;
  try {
    e.getByLabel(label_, instanceName_, digis);
  } catch ( cms::Exception &e ) {
    LogDebug("") << "ESOccupancy : Error! can't get collection !" << std::endl;
  }
  
  int occ[2][4][4];
  for (int i=0; i<2; ++i) 
    for (int j=0; j<4; ++j) 
      for (int k=0; k<4; ++k) 
	occ[i][j][k] = 0;

  for (ESDigiCollection::const_iterator digiItr = digis->begin(); digiItr != digis->end(); ++digiItr ) {

    ESDataFrame dataframe = (*digiItr);
    ESDetId id = dataframe.id();

    int plane = id.plane();
    int ix    = id.six();
    int iy    = id.siy();

    occ[plane-1][ix-30][iy-19]++;
  }

  for (int i=0; i<2; ++i) 
    for (int j=0; j<4; ++j) 
      for (int k=0; k<4; ++k) 
	meOccupancy_[i][j][k]->Fill(occ[i][j][k]);
  
  sleep(20);
}

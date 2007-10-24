#include "DQM/EcalPreshowerMonitorModule/interface/ESOccupancyCTTask.h"

#include <iostream>

using namespace cms;
using namespace edm;
using namespace std;

ESOccupancyCTTask::ESOccupancyCTTask(const ParameterSet& ps) {

  digilabel_    = ps.getUntrackedParameter<string>("DigiLabel");
  rechitlabel_  = ps.getUntrackedParameter<string>("RecHitLabel");
  instanceName_ = ps.getUntrackedParameter<string>("InstanceES");
  gain_         = ps.getUntrackedParameter<int>("ESGain", 1);
  sta_          = ps.getUntrackedParameter<bool>("RunStandalone", false);

  init_ = false;

  for (int i=0; i<2; ++i) {
    for (int j=0; j<6; ++j) {
      meEnergy_[i][j] = 0;
      meOccupancy1D_[i][j] = 0;  
      meOccupancy2D_[i][j] = 0;  
    }
  }

}

ESOccupancyCTTask::~ESOccupancyCTTask(){
}

void ESOccupancyCTTask::beginJob(const EventSetup& c) {
  
  ievt_ = 0;
  
  dbe_ = Service<DaqMonitorBEInterface>().operator->();
  
  if ( dbe_ ) {
    dbe_->setCurrentFolder("ES/ESOccupancyCTTask");
    dbe_->rmdir("ES/ESOccupancyCTTask");
  }
  
}

void ESOccupancyCTTask::setup(void){
  
  init_ = true;
  
  Char_t hist[200];
  
  dbe_ = Service<DaqMonitorBEInterface>().operator->();
  
  if ( dbe_ ) {   
    dbe_->setCurrentFolder("ES/ESOccupancyCTTask");
    for (int i=0; i<2; ++i) {       
      for (int j=0; j<6; ++j) {

	sprintf(hist, "ES Energy Box %d P %d", i+1, j+1);      
	meEnergy_[i][j] = dbe_->book1D(hist, hist, 500, 0, 500);

	sprintf(hist, "ES Occupancy 1D Box %d P %d", i+1, j+1);      
	meOccupancy1D_[i][j] = dbe_->book1D(hist, hist, 30, 0, 30);

	sprintf(hist, "ES Occupancy 2D Box %d P %d", i+1, j+1);      
	meOccupancy2D_[i][j] = dbe_->book2D(hist, hist, 64, 0, 64, 5, 0, 5);
      }
    }
  }

}

void ESOccupancyCTTask::cleanup(void){
  
  if (sta_) return;
  
  dbe_ = Service<DaqMonitorBEInterface>().operator->();
  
  if ( dbe_ ) {
    dbe_->setCurrentFolder("ES/ESOccupancyCTTask");
    
    for (int i=0; i<2; ++i) {
      for (int j=0; j<6; ++j) {
	if ( meEnergy_[i][j] ) dbe_->removeElement( meEnergy_[i][j]->getName() );
	meEnergy_[i][j] = 0;
	
	if ( meOccupancy1D_[i][j] ) dbe_->removeElement( meOccupancy1D_[i][j]->getName() );
	meOccupancy1D_[i][j] = 0;

	if ( meOccupancy2D_[i][j] ) dbe_->removeElement( meOccupancy2D_[i][j]->getName() );
	meOccupancy2D_[i][j] = 0;
      }
    }
  }

  init_ = false;
}

void ESOccupancyCTTask::endJob(void) {

  LogInfo("ESOccupancyCTTask") << "analyzed " << ievt_ << " events";

  if ( init_ ) this->cleanup();
}

void ESOccupancyCTTask::analyze(const Event& e, const EventSetup& c){

  if ( ! init_ ) this->setup();
  ievt_++;

  Handle<ESRawDataCollection> dccs;
  try {
    e.getByLabel(digilabel_, dccs);
  } catch ( cms::Exception &e ) {
    LogDebug("") << "ESOccupancy : Error! can't get ES raw data collection !" << std::endl;
  }

  //Handle<ESDigiCollection> digis;
  //try {
  //e.getByLabel(label_, instanceName_, digis);
  //} catch ( cms::Exception &e ) {
  //LogDebug("") << "ESOccupancy : Error! can't get collection !" << std::endl;
  //}

  Handle<ESRecHitCollection> hits;
  try {
    e.getByLabel(rechitlabel_, instanceName_, hits);
  } catch ( cms::Exception &e ) {
    LogDebug("") << "ESOccupancy : Error! can't get ES rec hit collection !" << std::endl;
  }

  // DCC
  vector<int> tdc;
  for ( ESRawDataCollection::const_iterator dccItr = dccs->begin(); dccItr != dccs->end(); ++dccItr ) {
    
    ESDCCHeaderBlock dcc = (*dccItr);
     
    if (dcc.fedId()==4) tdc = dcc.getTDCChannel();   
  }

  double fts = ((double)tdc[7]-800.)*24.951/194.;
  
  double w[3];
  if (gain_==1) {
    w[0] = -0.35944  + 0.03199 * fts;
    w[1] =  0.58562  + 0.03724 * fts;
    w[2] =  0.77204  - 0.06913 * fts;
  } else {
    w[0] = -0.26888  + 0.01807 * fts;
    w[1] =  0.54452  + 0.03204 * fts;
    w[2] =  0.72597  - 0.05062 * fts;
  }

  double zsThreshold = 3.*sqrt(w[0]*w[0]+w[1]*w[1]+w[2]*w[2])*7.*81.08/50./1000000.;
  //cout<<"ZS : "<<zsThreshold<<" "<<tdc[7]<<" "<<fts<<" "<<sqrt(w[0]*w[0]+w[1]*w[1]+w[2]*w[2])<<" "<<w[0]<<" "<<w[1]<<" "<<w[2]<<endl;

  //for (ESDigiCollection::const_iterator digiItr = digis->begin(); digiItr != digis->end(); ++digiItr ) {
  //ESDataFrame dataframe = (*digiItr);
  //ESDetId id = dataframe.id();
  //int plane = id.plane();
  //int ix    = id.six();
  //int iy    = id.siy();
  //}

  int occ[2][6];
  for (int m=0; m<2; ++m) 
    for (int n=0; n<6; ++n) 
      occ[m][n] = 0;  

  for ( ESRecHitCollection::const_iterator recHit = hits->begin(); recHit != hits->end(); ++recHit )  {

    if (recHit->energy()<zsThreshold) continue;

    ESDetId ESid = ESDetId(recHit->id());
    
    //int zside = ESid.zside();
    //int plane = ESid.plane();
    int ix    = ESid.six();
    int iy    = ESid.siy();
    int strip = ESid.strip();
    
    int j = (ix-1)/2; 
    int i;
    if (j<=5) i = 0;
    else i = 1;
    if (j>5) j = j-6;    
    int k = (ix-1)%2;

    meEnergy_[i][j]->Fill(recHit->energy()*1000000.);
    meOccupancy2D_[i][j]->Fill(abs(strip-32-k*32), iy-1, 1);

    occ[i][j]++;
  }

  for (int m=0; m<2; ++m) 
    for (int n=0; n<6; ++n) 
      meOccupancy1D_[m][n]->Fill(occ[m][n]);
  
}

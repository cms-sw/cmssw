#include "DQM/EcalPreshowerMonitorModule/interface/ESTDCCTTask.h"

#include <iostream>

using namespace cms;
using namespace edm;
using namespace std;

ESTDCCTTask::ESTDCCTTask(const ParameterSet& ps) {
  
  label_        = ps.getUntrackedParameter<string>("Label");
  instanceName_ = ps.getUntrackedParameter<string>("InstanceES");
  pedestalFile_ = ps.getUntrackedParameter<string>("PedestalFile");
  sta_          = ps.getUntrackedParameter<bool>("RunStandalone", false);
  
  init_ = false;
  
  ped_ = new TFile(pedestalFile_.c_str());  //Root file with ped histos
  Char_t tmp[300];  

  meTDC_ = 0;
  meEloss_ = 0;

  for (int i=0; i<2; ++i) 
    for (int j=0; j<6; ++j)
      for (int k=0; k<3; ++k) 
	meTDCADC_[i][j][k] = 0;
  
  for (int i=0; i<2; ++i){ 
    for (int j=0; j<6; ++j){
      for (int k=0; k<2; ++k){        
        for (int m=0; m<5; ++m){
	  int zside = (i==0)?1:-1;
	  sprintf(tmp,"DQMData/ES/QT/PedestalCT/ES Pedestal Fit Mean RMS Z %d P %1d Row %02d Col %02d",zside,j+1,k+1,m+1);
	  hist_[i][j][k][m] = (TH1F*) ped_->Get(tmp);
	}
      }
    }
  }

  dbe_ = Service<DaqMonitorBEInterface>().operator->();
  
}

ESTDCCTTask::~ESTDCCTTask(){
}

void ESTDCCTTask::beginJob(const EventSetup& c) {
  
  ievt_ = 0;
  
  if ( dbe_ ) {
    dbe_->setCurrentFolder("ES/ESTDCCTTask");
    dbe_->rmdir("ES/ESTDCCTTask");
  }

}

void ESTDCCTTask::setup(void){
  
  init_ = true;
  
  Char_t hist[200];
  
  if ( dbe_ ) {   
    dbe_->setCurrentFolder("ES/ESTDCCTTask");
    
    sprintf(hist, "ES TDC");
    meTDC_ = dbe_->book1D(hist, hist, 120, 520, 640);

    sprintf(hist, "ES Eloss");
    meEloss_ = dbe_->book1D(hist, hist, 60, 0, 300);

    for (int i=0; i<2; ++i) {

      int zside = (i==0)?1:-1;

      for (int j=0; j<6; ++j) {

	sprintf(hist, "ES TDC ADC Z %d P %d", zside, j+1);
	meTDCADCT_[i][j] = dbe_->book2D(hist, hist, 75, 0, 75, 400, -100, 300);
	
	for (int k=0; k<3; ++k) { 
	  sprintf(hist, "ES TDC ADC Z %d P %d Slot %d", zside, j+1, k+1);
	  meTDCADC_[i][j][k] = dbe_->book2D(hist, hist, 120, 520, 640, 400, -100, 300);
	}
      }
    }
   }
  
}

void ESTDCCTTask::cleanup(void){

  if (sta_) return;

  if ( dbe_ ) {
    dbe_->setCurrentFolder("ES/ESTDCCTTask");
    
    if ( meTDC_ ) dbe_->removeElement( meTDC_->getName() );
    meTDC_ = 0;

    if ( meEloss_ ) dbe_->removeElement( meEloss_->getName() );
    meEloss_ = 0;

    for (int i=0; i<2; ++i) {
      for (int j=0; j<6; ++j) {
	if ( meTDCADCT_[i][j] ) dbe_->removeElement( meTDCADCT_[i][j]->getName() );
	meTDCADCT_[i][j] = 0;
	for (int k=0; k<3; ++k) { 
	  if ( meTDCADC_[i][j][k] ) dbe_->removeElement( meTDCADC_[i][j][k]->getName() );
	  meTDCADC_[i][j][k] = 0;
	}
      }
    }
  }
 
  init_ = false;

}

void ESTDCCTTask::endJob(void) {

  LogInfo("ESTDCCTTask") << "analyzed " << ievt_ << " events";

  if ( init_ ) this->cleanup();

}

void ESTDCCTTask::analyze(const Event& e, const EventSetup& c){

  if ( ! init_ ) this->setup();
  ievt_++;
  
  Handle<ESRawDataCollection> dccs;
  try {
    e.getByLabel(label_, instanceName_, dccs);
  } catch ( cms::Exception &e ) {
    LogDebug("") << "ESTDCCT : Error! can't get ES raw data collection !" << std::endl;
  }  

  Handle<ESDigiCollection> digis;
  try {
    e.getByLabel(label_, instanceName_, digis);
  } catch ( cms::Exception &e ) {
    LogDebug("") << "ESTDCCT : Error! can't get digis collection !" << std::endl;
  } 

  // DCC
  vector<int> tdcStatus;
  vector<int> tdc;
  
  for ( ESRawDataCollection::const_iterator dccItr = dccs->begin(); dccItr != dccs->end(); ++dccItr ) {
    
    ESDCCHeaderBlock dcc = (*dccItr);
     
    if (dcc.fedId()==4) {       

      tdcStatus = dcc.getTDCChannelStatus();
      tdc = dcc.getTDCChannel();

      meTDC_->Fill(tdc[7]);
    }    
  }

  // Digis
  int ix, iy, strip, i, j, k, pedestal;

  for (ESDigiCollection::const_iterator digiItr = digis->begin(); digiItr != digis->end(); ++digiItr ) {
    
    ESDataFrame dataframe = (*digiItr);
    ESDetId id = dataframe.id();
    
    //int plane = id.plane();
    ix    = id.six();
    iy    = id.siy();
    strip = id.strip();

    j = (ix-1)/2; 
    if (j<=5) i = 0;
    else i = 1;
    if (j>5) j = j-6;    
    k = (ix-1)%2;

    float pedestal = hist_[i][j][k][iy-1]->GetBinContent(strip);

    for (int isam=0; isam<dataframe.size(); ++isam) {

      double sampleT = (tdc[7]-548)*24.951/74.+isam*24.951;

      meTDCADC_[i][j][isam]->Fill(tdc[7], dataframe.sample(isam).adc()-pedestal);
      meTDCADCT_[i][j]->Fill( sampleT, dataframe.sample(isam).adc()-pedestal);
      
      if (sampleT>49.5 && sampleT<50.5 && isam==1 && (dataframe.sample(isam).adc()-pedestal)>30.) 
	meEloss_->Fill(dataframe.sample(isam).adc()-pedestal);      
    }
    
  }


}

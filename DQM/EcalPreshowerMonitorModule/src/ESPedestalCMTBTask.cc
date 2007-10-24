#include "DQM/EcalPreshowerMonitorModule/interface/ESPedestalCMTBTask.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBTriggerData.h"

#include <iostream>

using namespace cms;
using namespace edm;
using namespace std;

ESPedestalCMTBTask::ESPedestalCMTBTask(const ParameterSet& ps) {

  label_        = ps.getUntrackedParameter<string>("Label");
  instanceName_ = ps.getUntrackedParameter<string>("InstanceES");
  pedestalFile_ = ps.getUntrackedParameter<string>("PedestalFile");
  sta_          = ps.getUntrackedParameter<bool>("RunStandalone", false);
  doCM_         = ps.getUntrackedParameter<bool>("DoCM", true);
  cmMethod_     = ps.getUntrackedParameter<int>("CMMethod", 1);
  sigma_        = ps.getUntrackedParameter<double>("SigmaForCM", 4);
  zs_           = ps.getUntrackedParameter<double>("ThreeSigmaForZS", 3.5);
  gain_         = ps.getUntrackedParameter<int>("ESGain", 1);

  init_ = false;

  ped_ = new TFile(pedestalFile_.c_str());  //Root file with ped histos
  Char_t tmp[300];

  for (int i=0; i<2; ++i){ 
    for (int j=0; j<4; ++j){
      for (int k=0; k<4; ++k){ 
       meSensorCM_S0_[i][j][k]=0;
       meSensorCM_S1_[i][j][k]=0;
       meSensorCM_S2_[i][j][k]=0;

       sprintf(tmp,"DQMData/ES/QT/PedestalTB/ES Pedestal Fit Mean RMS Z 1 P %1d Col %02d Row %02d",i+1,j+1,k+1);
       hist_[i][j][k] = (TH1F*) ped_->Get(tmp);

	for (int m=0; m<32; ++m){
	  mePedestalCM_S0_[i][j][k][m] = 0;  
	  mePedestalCM_S1_[i][j][k][m] = 0;  
	  mePedestalCM_S2_[i][j][k][m] = 0;  
        }
      }
    }
    
    for (int n=0; n<3; ++n) {
      meADC_[i][n] = 0;
      meADCZS_[i][n] = 0;
      meOccupancy2D_[i][n] = 0;
    }
  }

  dbe_ = Service<DaqMonitorBEInterface>().operator->();
}

ESPedestalCMTBTask::~ESPedestalCMTBTask(){

  delete ped_;

}

void ESPedestalCMTBTask::beginJob(const EventSetup& c) {
  
  ievt_ = 0;
  
  if ( dbe_ ) {
    dbe_->setCurrentFolder("ES/ESPedestalCMTBTask");
    dbe_->rmdir("ES/ESPedestalCMTBTask");
  }
  
}

void ESPedestalCMTBTask::setup(void){
  
  init_ = true;
  
  Char_t hist[200];
  
  if ( dbe_ ) {   
    dbe_->setCurrentFolder("ES/ESPedestalCMTBTask");
    for (int i=0; i<2; ++i) {       

      for (int m=0; m<3; ++m) {
	sprintf(hist, "ES ADC P %d TS %d", i+1, m+1);
	meADC_[i][m] = dbe_->book1D(hist, hist, 700, -200, 500);

	sprintf(hist, "ES ADC ZS P %d TS %d", i+1, m+1);
	meADCZS_[i][m] = dbe_->book1D(hist, hist, 700, 0, 700);

	sprintf(hist, "ES Occupancy P %d TS %d", i+1, m+1);
	if (i==0) meOccupancy2D_[i][m] = dbe_->book2D(hist, hist, 128, 0, 128, 4, 0, 4);
	else if (i==1) meOccupancy2D_[i][m] = dbe_->book2D(hist, hist, 4, 0, 4, 128, 0, 128);
      }

      for (int j=0; j<4; ++j) {
	for (int k=0; k<4; ++k) {
	 sprintf(hist, "ES Sensor CM_S0 Z 1 P %d Col %02d Row %02d", i+1, j+1, k+1);
	 meSensorCM_S0_[i][j][k] = dbe_->book1D(hist, hist, 400, -200, 200);	  

	 sprintf(hist, "ES Sensor CM_S1 Z 1 P %d Col %02d Row %02d", i+1, j+1, k+1);
	 meSensorCM_S1_[i][j][k] = dbe_->book1D(hist, hist, 400, -200, 200); 

	 sprintf(hist, "ES Sensor CM_S2 Z 1 P %d Col %02d Row %02d", i+1, j+1, k+1);
	 meSensorCM_S2_[i][j][k] = dbe_->book1D(hist, hist, 400, -200, 200);	  
	 
	 for (int m=0; m<32; ++m) {
	   sprintf(hist, "ES Pedestal CM_0 Z 1 P %d Col %02d Row %02d Str %02d", i+1, j+1, k+1, m+1);      
	   mePedestalCM_S0_[i][j][k][m] = dbe_->book1D(hist, hist, 5000, -200, 4800);

	   sprintf(hist, "ES Pedestal CM_1 Z 1 P %d Col %02d Row %02d Str %02d", i+1, j+1, k+1, m+1);      
	   mePedestalCM_S1_[i][j][k][m] = dbe_->book1D(hist, hist, 5000, -200, 4800);

	   sprintf(hist, "ES Pedestal CM_2 Z 1 P %d Col %02d Row %02d Str %02d", i+1, j+1, k+1, m+1);      
	   mePedestalCM_S2_[i][j][k][m] = dbe_->book1D(hist, hist, 5000, -200, 4800);
	 }
	}
      }
    }
  }

}

void ESPedestalCMTBTask::cleanup(void){

  if (sta_) return;

  if ( dbe_ ) {
    dbe_->setCurrentFolder("ES/ESPedestalCMTBTask");

    for (int i=0; i<2; ++i) {

      for (int m=0; m<3; ++m) {
	if ( meADC_[i][m] ) dbe_->removeElement( meADC_[i][m]->getName() );
	meADC_[i][m] = 0;

	if ( meADCZS_[i][m] ) dbe_->removeElement( meADCZS_[i][m]->getName() );
	meADCZS_[i][m] = 0;

	if ( meOccupancy2D_[i][m] ) dbe_->removeElement( meOccupancy2D_[i][m]->getName() );
	meOccupancy2D_[i][m] = 0;
      }

      for (int j=0; j<4; ++j) {
	for (int k=0; k<4; ++k) {
          if ( meSensorCM_S0_[i][j][k] ) dbe_->removeElement( meSensorCM_S0_[i][j][k]->getName() );
          meSensorCM_S0_[i][j][k] = 0;

          if ( meSensorCM_S1_[i][j][k] ) dbe_->removeElement( meSensorCM_S1_[i][j][k]->getName() );
          meSensorCM_S1_[i][j][k] = 0;

          if ( meSensorCM_S2_[i][j][k] ) dbe_->removeElement( meSensorCM_S2_[i][j][k]->getName() );
          meSensorCM_S2_[i][j][k] = 0;

	  for (int m=0; m<32; ++m) {
	    if ( mePedestalCM_S0_[i][j][k][m] ) dbe_->removeElement( mePedestalCM_S0_[i][j][k][m]->getName() );
	    mePedestalCM_S0_[i][j][k][m] = 0;

	    if ( mePedestalCM_S1_[i][j][k][m] ) dbe_->removeElement( mePedestalCM_S1_[i][j][k][m]->getName() );
	    mePedestalCM_S1_[i][j][k][m] = 0;

	    if ( mePedestalCM_S2_[i][j][k][m] ) dbe_->removeElement( mePedestalCM_S2_[i][j][k][m]->getName() );
	    mePedestalCM_S2_[i][j][k][m] = 0;
	  }
	}
      }
    }

  }

  init_ = false;

}

void ESPedestalCMTBTask::endJob(void) {

  LogInfo("ESPedestalCMTBTask") << "analyzed " << ievt_ << " events";

  if ( init_ ) this->cleanup();

}

void ESPedestalCMTBTask::DoCommonModeItr(float data[], float *cm) {

  // Fisrt loop
  float vAll = 0;
  float vAll2 = 0;
  int nstr = 0;
  int nstr1 = 0;
  for (int i=0; i<32; ++i) { vAll += data[i]; nstr++; }
  float mean = vAll/32.;
  for (int i=0; i<32; ++i) vAll2 += data[i]*data[i];
  float rms = sqrt(vAll2/32.);

  int count = 0;

  // Looping
  while (nstr1 != nstr) {
    vAll = 0;
    vAll2 = 0;
    if (count!=0) nstr = nstr1;
    nstr1 = 0;    
    for (int i=0; i<32; ++i) {
      if (fabs(data[i]-mean) < 3.*rms) {
	vAll += data[i];
	nstr1++;
      }
    }
    mean = vAll/nstr1;
    for (int i=0; i<32; ++i) 
      if (fabs(data[i]-mean) < 3.*rms) 
	vAll2 += data[i]*data[i];
    rms = sqrt(vAll2/nstr1);

    //cout<<"CM : "<<count<<" "<<nstr<<" "<<nstr1<<" "<<mean<<" "<<rms<<endl;
    count++;
  }

  *cm = mean;

}

void ESPedestalCMTBTask::DoCommonMode32(float det_data[], float *cm) {

  int current_min1 = 4095;
  int current_strip_val1 = 0;
  int current_sum1 = 0;
  int threshold1=0;
  int n1=0;
  int check_bit1_10;
  int corrected_sum1;

  int treshold_const = (int) 6.*sigma_;  //+-3sigma?

  int myltip_factor[]={2048, 1024, 683, 512, 410, 341, 293, 256, 228, 205, 186, 171, 158, 146, 137, 128, 
		       120, 114, 108, 102, 98, 93, 89, 85, 82, 79, 76, 73, 71, 68, 66, 64};

  for(int i=0; i<32; ++i){                    //Evaluate min value strip 0-15
    current_strip_val1 = (int) det_data[i];
    if(current_min1 > current_strip_val1 ) {
      current_min1 = current_strip_val1;
    }
  } 
  threshold1 = current_min1 + treshold_const;
  
  for(int i=0; i<32; ++i){                    //Sum all noisy strips 0-15
    current_strip_val1 = (int) det_data[i];
    if(current_strip_val1 <= threshold1 ) {
      n1++;
      current_sum1 += current_strip_val1;
    }
  } 
  corrected_sum1 = current_sum1*myltip_factor[n1-1];   // multiply sum by factor
  corrected_sum1 = corrected_sum1 >> 10;               //shilft right 10 bits [9:0]
  check_bit1_10 = corrected_sum1 & 1;                  //check bit [10]
  corrected_sum1 = corrected_sum1 >> 1;                //shilft right 1 more bit [10]
  if(check_bit1_10 == 1) corrected_sum1++;             //increase by 1 if bit [10] was one
  
  *cm = (float)corrected_sum1;
  
}

void ESPedestalCMTBTask::DoCommonMode(float det_data[], float *cm1, float *cm2) {

   int current_min1 = 4095;
   int current_strip_val1 = 0;
   int current_sum1 = 0;
   int threshold1=0;
   int n1=0;
   int check_bit1_10;
   int corrected_sum1;

   int current_min2 = 4095;
   int current_strip_val2 = 0;
   int current_sum2 = 0;
   int threshold2=0;
   int n2=0;
   int check_bit2_10;
   int corrected_sum2;

   int treshold_const = (int) 8.*sigma_;  //+-3sigma?
   int myltip_factor[]={2048, 1024, 683, 512, 410, 341, 293, 256, 228, 205,
                        186, 171, 158, 146, 137, 128};

   // Common mode for strips 0-15
   
   for(int i=0; i<16; ++i){                    //Evaluate min value strip 0-15
     current_strip_val1 = (int) det_data[i];
     if(current_min1 > current_strip_val1 ) {
       current_min1 = current_strip_val1;
     }
   } 
   threshold1 = current_min1 + treshold_const;

   for(int i=0; i<16; ++i){                    //Sum all noisy strips 0-15
     current_strip_val1 = (int) det_data[i];
     if(current_strip_val1 <= threshold1 ) {
       n1++;
       current_sum1 += current_strip_val1;
     }
   } 
   corrected_sum1 = current_sum1*myltip_factor[n1-1];   // multiply sum by factor
   corrected_sum1 = corrected_sum1 >> 10;               //shilft right 10 bits [9:0]
   check_bit1_10 = corrected_sum1 & 1;                  //check bit [10]
   corrected_sum1 = corrected_sum1 >> 1;                //shilft right 1 more bit [10]
   if(check_bit1_10 == 1) corrected_sum1++;             //increase by 1 if bit [10] was one

   *cm1=(float)corrected_sum1;
   // *cm1=current_sum1/n1;

   // Common mode for strips 16-31
   
   for(int i=16; i<32; ++i){                    //Evaluate min value strip 16-31
     current_strip_val2 = (int) det_data[i];
     if(current_min2 > current_strip_val2) {
       current_min2 = current_strip_val2;
     }
   } 
   threshold2 = current_min2 + treshold_const;

   for(int i=16; i<32; ++i){                    //Sum all noisy strips 16-31
     current_strip_val2 = (int) det_data[i];
     if(current_strip_val2 <= threshold2 ) {
       n2++;
       current_sum2 += current_strip_val2;
     }
   } 
   corrected_sum2 = current_sum2*myltip_factor[n2-1];   // multiply sum by factor
   corrected_sum2 = corrected_sum2 >> 10;               //shilft right 10 bits [9:0]
   check_bit2_10 = corrected_sum2 & 1;                  //check bit [10]
   corrected_sum2 = corrected_sum2 >> 1;                //shilft right 1 more bit [10]
   if(check_bit2_10 == 1) corrected_sum2++;             //increase by 1 if bit [10] was one

   *cm2=(float)corrected_sum2;   
   // *cm2=current_sum2/n2;

}

void ESPedestalCMTBTask::analyze(const Event& e, const EventSetup& c){

  if ( ! init_ ) this->setup();
  ievt_++;
  
  Handle<HcalTBTriggerData> trg;
  try {
    e.getByType(trg);
  }
  catch ( cms::Exception &e ) {
    LogDebug("") << "ESPedestal : Error! can't get trigger information !" << std::endl;
  }

  int trgbit = 0;
  if( trg->wasBeamTrigger() )             trgbit = 1;
  if( trg->wasInSpillPedestalTrigger() )  trgbit = 2;
  if( trg->wasOutSpillPedestalTrigger() ) trgbit = 3;
  if( trg->wasLEDTrigger() )              trgbit = 4;
  if( trg->wasLaserTrigger() )            trgbit = 5;

  Handle<ESDigiCollection> digis;
  try {
    e.getByLabel(label_, instanceName_, digis);
  } catch ( cms::Exception &e ) {
    LogDebug("") << "ESPedestal : Error! can't get collection !" << std::endl;
  } 

  //Need for storing original data 
  int data_S0[2][4][4][32];
  int data_S1[2][4][4][32];
  int data_S2[2][4][4][32];
  for (int i=0; i<2; ++i) 
    for (int j=0; j<4; ++j)
      for (int k=0; k<4; ++k)        
	for (int m=0; m<32; ++m) {
	  data_S0[i][j][k][m] = 0;
	  data_S1[i][j][k][m] = 0;
	  data_S2[i][j][k][m] = 0;
	}

  //Need for storing data after CM correction 
  int dataCM_S0[2][4][4][32];
  int dataCM_S1[2][4][4][32];
  int dataCM_S2[2][4][4][32];
  for (int i=0; i<2; ++i) 
    for (int j=0; j<4; ++j)
      for (int k=0; k<4; ++k)        
	for (int m=0; m<32; ++m) {
	  dataCM_S0[i][j][k][m] = 0;
	  dataCM_S1[i][j][k][m] = 0;
	  dataCM_S2[i][j][k][m] = 0;
	}

  for (ESDigiCollection::const_iterator digiItr = digis->begin(); digiItr != digis->end(); ++digiItr ) {

    ESDataFrame dataframe = (*digiItr);
    ESDetId id = dataframe.id();

    int plane = id.plane();
    int ix    = id.six();
    int iy    = id.siy();
    int strip = id.strip();

    //printf("plane=%d ix=%d iy=%d strip=%d\n",plane, ix, iy, strip);
    //printf("mean_strip_pedestal=%d \n",mean_strip_pedestal);

    data_S0[plane-1][ix-1][iy-1][strip-1]=dataframe.sample(0).adc();
    data_S1[plane-1][ix-1][iy-1][strip-1]=dataframe.sample(1).adc();
    data_S2[plane-1][ix-1][iy-1][strip-1]=dataframe.sample(2).adc();

  }

  float sensor_data_S0[32];
  float sensor_data_S1[32];
  float sensor_data_S2[32];
  float com_mode;
  float com_mode1;
  float com_mode2;
  float pedrms = 0;
  int avgcount = 0;

  for (int i=0; i<2; ++i) { 
    for (int j=0; j<4; ++j) {
      for (int k=0; k<4; ++k) {   

	//cout<<"Sensor : "<<i+1<<" "<<j+1<<" "<<k+1<<" "<<hist_[i][j][k]->GetEntries()<<endl;
	avgcount = 0;     
	pedrms = 0;

	for (int m=0; m<32; ++m) {

	  sensor_data_S0[m]=data_S0[i][j][k][m];  //Read sensor data
	  sensor_data_S1[m]=data_S1[i][j][k][m];  //Read sensor data
	  sensor_data_S2[m]=data_S2[i][j][k][m];  //Read sensor data

	  // dead channel
	  //if (i==1 && j==0 && k==2 && m==3) {
	  if (i==1 && j==0 && k==3 && m==3) {
	    sensor_data_S0[m] = 4095;
	    sensor_data_S1[m] = 4095;
	    sensor_data_S2[m] = 4095;
	  }

	  if (i==1 && j==0 && k==3 && m==3) continue;
	  pedrms += hist_[i][j][k]->GetBinError(m+1);
	  avgcount++;
	}
	
	//printf("******************************************\n"); 
	//for(int kk=0; kk<32; ++kk){printf("%6.0f",sensor_data[kk]);}
	//printf("\n");

	if (hist_[i][j][k]->GetEntries() == 0) continue;	

	for(int kk=0; kk<32; ++kk){
	  int pedestal = (int)hist_[i][j][k]->GetBinContent(kk+1);
	  //float pedestal=1001;
	  //printf("GaussMean=%f\n",pedestal);
	  
	  sensor_data_S0[kk]=sensor_data_S0[kk]-pedestal;   //Pedestal subtraction
	  sensor_data_S1[kk]=sensor_data_S1[kk]-pedestal;   //Pedestal subtraction
	  sensor_data_S2[kk]=sensor_data_S2[kk]-pedestal;   //Pedestal subtraction
	}
			
	// correct for common mode
	if (cmMethod_ == 1) {
	  DoCommonMode(sensor_data_S0, &com_mode1,&com_mode2);  //Common mode calculation
	  meSensorCM_S0_[i][j][k]->Fill(com_mode1);  //Fill CM histos per sensor
	  meSensorCM_S0_[i][j][k]->Fill(com_mode2);
	} else if (cmMethod_ == 2) {
	  DoCommonMode32(sensor_data_S0, &com_mode);  //Common mode calculation
	  meSensorCM_S0_[i][j][k]->Fill(com_mode);  //Fill CM histos per sensor
	}
	//printf("cm1=%f cm2=%f\n",com_mode1,com_mode2);

	if (!doCM_) {
	  com_mode1 = 0;
	  com_mode2 = 0;
	  com_mode  = 0;
	}
	for(int kk=0; kk<16; ++kk){
	  if (cmMethod_ == 1) {
	    sensor_data_S0[kk]    -= com_mode1;  //Common mode correction 0:15
	    sensor_data_S0[kk+16] -= com_mode2;  //Common mode correction 16:31
	  } else if (cmMethod_ == 2) {
	    sensor_data_S0[kk]    -= com_mode;   //Common mode correction 0:15
	    sensor_data_S0[kk+16] -= com_mode;   //Common mode correction 16:31
	  }
	}

	if (cmMethod_ == 1) {
	  DoCommonMode(sensor_data_S1, &com_mode1,&com_mode2);  //Common mode calculation
	  meSensorCM_S1_[i][j][k]->Fill(com_mode1);  //Fill CM histos per sensor
	  meSensorCM_S1_[i][j][k]->Fill(com_mode2);
	} else if (cmMethod_ == 2) {
	  DoCommonMode32(sensor_data_S1, &com_mode);  //Common mode calculation
	  meSensorCM_S1_[i][j][k]->Fill(com_mode);  //Fill CM histos per sensor
	}
	//printf("cm1=%f cm2=%f\n",com_mode1,com_mode2);

	if (!doCM_) {
	  com_mode1 = 0;
	  com_mode2 = 0;
	  com_mode  = 0;
	}
	for(int kk=0; kk<16; ++kk){
	  if (cmMethod_ == 1) {
	    sensor_data_S1[kk]    -= com_mode1;  //Common mode correction 0:15
	    sensor_data_S1[kk+16] -= com_mode2;  //Common mode correction 16:31
	  } else if (cmMethod_ == 2) {
	    sensor_data_S1[kk]    -= com_mode;   //Common mode correction 0:15
	    sensor_data_S1[kk+16] -= com_mode;   //Common mode correction 16:31
	  }
	}

	if (cmMethod_ == 1) {
	  DoCommonMode(sensor_data_S2, &com_mode1,&com_mode2);  //Common mode calculation
	  meSensorCM_S2_[i][j][k]->Fill(com_mode1);  //Fill CM histos per sensor
	  meSensorCM_S2_[i][j][k]->Fill(com_mode2);
	} else if (cmMethod_ == 2) {
	  DoCommonMode32(sensor_data_S2, &com_mode);  //Common mode calculation
	  meSensorCM_S2_[i][j][k]->Fill(com_mode);  //Fill CM histos per sensor
	}
	//printf("cm1=%f cm2=%f\n",com_mode1,com_mode2);

	if (!doCM_) {
	  com_mode1 = 0;
	  com_mode2 = 0;
	  com_mode  = 0;
	}
	for(int kk=0; kk<16; ++kk){
	  if (cmMethod_ == 1) {
	    sensor_data_S2[kk]    -= com_mode1;  //Common mode correction 0:15
	    sensor_data_S2[kk+16] -= com_mode2;  //Common mode correction 16:31
	  } else if (cmMethod_ == 2) {
	    sensor_data_S2[kk]    -= com_mode;   //Common mode correction 0:15
	    sensor_data_S2[kk+16] -= com_mode;   //Common mode correction 16:31
	  }
	}

	for(int kk=0; kk<32; ++kk){
	  dataCM_S0[i][j][k][kk]=(int)sensor_data_S0[kk];  //Storing corrected data
	  dataCM_S1[i][j][k][kk]=(int)sensor_data_S1[kk];  //Storing corrected data
	  dataCM_S2[i][j][k][kk]=(int)sensor_data_S2[kk];  //Storing corrected data
	}
	
      }
    }
  }
  
  for (int i=0; i<2; ++i){ 
    for (int j=0; j<4; ++j){
      for (int k=0; k<4; ++k){        
	for (int m=0; m<32; ++m){

	  mePedestalCM_S0_[i][j][k][m]->Fill(dataCM_S0[i][j][k][m]);  //Filling histos with corrected data
	  mePedestalCM_S1_[i][j][k][m]->Fill(dataCM_S1[i][j][k][m]);  //Filling histos with corrected data
	  mePedestalCM_S2_[i][j][k][m]->Fill(dataCM_S2[i][j][k][m]);  //Filling histos with corrected data

	  // missing sensors
	  if (i==0 && j==0 && k==3) continue;
	  if (i==0 && j==3 && k==3) continue;

	  // dead channel
	  //if (i==1 && j==0 && k==2 && m==3) continue;
	  if (i==1 && j==0 && k==3 && m==3) continue;

	  if (trgbit != 1) continue;

	  meADC_[i][0]->Fill(dataCM_S0[i][j][k][m]);
	  meADC_[i][1]->Fill(dataCM_S1[i][j][k][m]);
	  meADC_[i][2]->Fill(dataCM_S2[i][j][k][m]);

	  if (dataCM_S0[i][j][k][m]>zs_) {
	    meADCZS_[i][0]->Fill(dataCM_S0[i][j][k][m]);
	    if (i==0) meOccupancy2D_[i][0]->Fill(abs(m-31)+32*j, k, 1);
	    if (i==1) meOccupancy2D_[i][0]->Fill(j, k*32+m, 1);
	  }

	  if (dataCM_S1[i][j][k][m]>zs_) {
	    meADCZS_[i][1]->Fill(dataCM_S1[i][j][k][m]);
	    if (i==0) meOccupancy2D_[i][1]->Fill(abs(m-31)+32*j, k, 1);
	    if (i==1) meOccupancy2D_[i][1]->Fill(j, k*32+m, 1);
	  }

	  if (dataCM_S2[i][j][k][m]>zs_) {
	    meADCZS_[i][2]->Fill(dataCM_S2[i][j][k][m]);
	    if (i==0) meOccupancy2D_[i][2]->Fill(abs(m-31)+32*j, k, 1);
	    if (i==1) meOccupancy2D_[i][2]->Fill(j, k*32+m, 1);
	  }

	}
      }
    }
  }
  
}

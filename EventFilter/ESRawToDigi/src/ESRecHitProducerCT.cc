#include "EventFilter/ESRawToDigi/interface/ESRecHitProducerCT.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/Common/interface/EDCollection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Selector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TBDataFormats/ESTBRawData/interface/ESDCCHeaderBlock.h"
#include "TBDataFormats/ESTBRawData/interface/ESKCHIPBlock.h"
#include "TBDataFormats/ESTBRawData/interface/ESLocalRawDataCollections.h"
#include "TBDataFormats/ESTBRawData/interface/ESRawDataCollections.h"

ESRecHitProducerCT::ESRecHitProducerCT(ParameterSet const& ps)
{
  digiCollection_ = ps.getParameter<edm::InputTag>("ESdigiCollection");
  rechitCollection_ = ps.getParameter<std::string>("ESrechitCollection");
  produces<ESRecHitCollection>(rechitCollection_);
  
  gain_ = ps.getUntrackedParameter<int>("ESGain", 1);
  double ESMIPADC = ps.getUntrackedParameter<double>("ESMIPADC", 9);
  double ESMIPkeV = ps.getUntrackedParameter<double>("ESMIPkeV", 81.08);

  pedestalFile_ = ps.getUntrackedParameter<string>("PedestalFile");
  // 1 : CT, 2 : TB
  detType_      = ps.getUntrackedParameter<int>("DetectorType", 1);
  doCM_         = ps.getUntrackedParameter<bool>("doCM", true);

  ped_ = new TFile(pedestalFile_.c_str());  //Root file with ped histos
  Char_t tmp[300];

  if (detType_ == 1) {
    for (int i=0; i<2; ++i) { // zisde
      for (int j=0; j<6; ++j) { // plane
	for (int k=0; k<2; ++k) { // ix
	  for (int m=0; m<5; ++m) { // iy
	    int zside = (i==0)?1:-1;
	    sprintf(tmp,"DQMData/ES/QT/PedestalCT/ES Pedestal Fit Mean RMS Z %d P %1d Row %02d Col %02d",zside,j+1,k+1,m+1);
	    hist_[i][j][k][m] = (TH1F*) ped_->Get(tmp);	    
	  }	        
	}
      }
    }    
  } else if (detType_ == 2) {    
    for (int j=0; j<2; ++j) { // plane
      for (int k=0; k<4; ++k) { // ix
	for (int m=0; m<4; ++m) { // iy	    
	  sprintf(tmp,"DQMData/ES/QT/PedestalCT/ES Pedestal Fit Mean RMS Z 1 P %1d Row %02d Col %02d",j+1,k+1,m+1);
	  hist_[0][j][k][m] = (TH1F*) ped_->Get(tmp);	    
	}          
      }
    }
  }

  algo_ = new ESRecHitSimAlgoCT(gain_, ESMIPADC, ESMIPkeV); 
}

ESRecHitProducerCT::~ESRecHitProducerCT() {
  delete algo_;
  delete ped_;
}

void ESRecHitProducerCT::DoCommonMode(double det_data[], double *cm1, double *cm2) {

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


   int treshold_const = 24;  //+-3sigma?
   int myltip_factor[]={2048, 1024, 683, 512, 410, 341, 293, 256, 228, 205,
                        186, 171, 158, 146, 137, 128};


   // Common mode for strips 0-15
   
   for(int i=0; i<16; ++i){                    //Evaluate min value strip 0-15
     current_strip_val1 = (int) det_data[i];
     if(current_min1 > current_strip_val1) {
       current_min1 = current_strip_val1;
     }
   } 
   threshold1 = current_min1 + treshold_const;

   for(int i=0; i<16; ++i){                    //Sum all noisy strips 0-15
     current_strip_val1 = (int) det_data[i];
     if(current_strip_val1 <= threshold1) {
       n1++;
       current_sum1 += current_strip_val1;
     }
   } 
   corrected_sum1 = current_sum1*myltip_factor[n1-1];   // multiply sum by factor
   corrected_sum1 = corrected_sum1 >> 10;               //shilft right 10 bits [9:0]
   check_bit1_10 = corrected_sum1 & 1;                  //check bit [10]
   corrected_sum1 = corrected_sum1 >> 1;                //shilft right 1 more bit [10]
   if(check_bit1_10 == 1) corrected_sum1++;             //increase by 1 if bit [10] was one

   *cm1=(double)corrected_sum1;

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
     if(current_strip_val2 <= threshold2) {
       n2++;
       current_sum2 += current_strip_val2;
     }
   } 
   corrected_sum2 = current_sum2*myltip_factor[n2-1];   // multiply sum by factor
   corrected_sum2 = corrected_sum2 >> 10;               //shilft right 10 bits [9:0]
   check_bit2_10 = corrected_sum2 & 1;                  //check bit [10]
   corrected_sum2 = corrected_sum2 >> 1;                //shilft right 1 more bit [10]
   if(check_bit2_10 == 1) corrected_sum2++;             //increase by 1 if bit [10] was one

   *cm2=(double)corrected_sum2;

}

void ESRecHitProducerCT::produce(Event& e, const EventSetup& es) {
  
  Handle<ESRawDataCollection> dccs;
  try {
    e.getByLabel(digiCollection_, dccs);
  } catch ( cms::Exception &e ) {
    LogDebug("") << "ESTDCCT : Error! can't get ES raw data collection !" << std::endl;
  }  

  Handle<ESDigiCollection> digiHandle;  
  const ESDigiCollection* digi = 0;
  e.getByLabel( digiCollection_, digiHandle);
  digi = digiHandle.product();
  
  LogInfo("ESRecHitInfo") << "total # ESdigis: " << digi->size() ;  
  auto_ptr<ESRecHitCollection> rec(new EcalRecHitCollection());
  
  // DCC
  vector<int> tdcStatus;
  vector<int> tdc;
  
  for ( ESRawDataCollection::const_iterator dccItr = dccs->begin(); dccItr != dccs->end(); ++dccItr ) {
    
    ESDCCHeaderBlock dcc = (*dccItr);
    
    if (dcc.fedId()==4) {      
      tdcStatus = dcc.getTDCChannelStatus();
      tdc = dcc.getTDCChannel();      
    }    
  }  

  // Digis
  ESDigiCollection::const_iterator idigi;
  int i, j, k;

  ESDataFrame dataframe;
  ESDetId id;
  int plane, ix, iy, strip;
  
  //Need for storing original data 
  int data_S0[2][6][4][5][32];
  int data_S1[2][6][4][5][32];
  int data_S2[2][6][4][5][32];
  for (int i=0; i<2; ++i) 
    for (int j=0; j<6; ++j)
      for (int k=0; k<4; ++k)        
	for (int m=0; m<5; ++m)
	  for (int n=0; n<32; ++n){
	    data_S0[i][j][k][m][n] = 0;
	    data_S1[i][j][k][m][n] = 0;
	    data_S2[i][j][k][m][n] = 0;
          }
  
  
  //Need for storing data after CM correction                                                                                 
  int dataCM_S0[2][6][4][5][32];
  int dataCM_S1[2][6][4][5][32];
  int dataCM_S2[2][6][4][5][32];
  for (int i=0; i<2; ++i)
    for (int j=0; j<6; ++j)
      for (int k=0; k<4; ++k)
        for (int m=0; m<5; ++m)
          for (int n=0; n<32; ++n) {
            dataCM_S0[i][j][k][m][n] = 0;
            dataCM_S1[i][j][k][m][n] = 0;
            dataCM_S2[i][j][k][m][n] = 0;
          }

  //Need for storing data of CMN
  double CMN_S0[2][6][4][5][2];
  double CMN_S1[2][6][4][5][2];
  double CMN_S2[2][6][4][5][2];
  for (int i=0; i<2; ++i)
    for (int j=0; j<6; ++j)
      for (int k=0; k<4; ++k)
        for (int m=0; m<5; ++m)
          for (int n=0; n<2; ++n) {
            CMN_S0[i][j][k][m][n] = 0;
            CMN_S1[i][j][k][m][n] = 0;
            CMN_S2[i][j][k][m][n] = 0;
          }

  // Store data for CM calculation
  for (idigi=digi->begin(); idigi!=digi->end(); ++idigi) { 

    dataframe = (*idigi);
    id = dataframe.id();
    plane = id.plane();
    ix    = id.six();
    iy    = id.siy();
    strip = id.strip();    

    if (detType_ == 1) {
      j = (ix-1)/2;
      if (j<=5) i = 0;
      else i = 1;
      if (j>5) j=j-6;
      k = (ix-1)%2;
    } else if (detType_ == 2) {
      i = 0;
      j = plane-1;
      k = ix-1; 
    }

    if (ix==7 && iy==5 && strip==4) {
      data_S0[i][j][k][iy-1][strip-1] = 2000.; //storing S0 data
      data_S1[i][j][k][iy-1][strip-1] = 2000.; //storing S1 data
      data_S2[i][j][k][iy-1][strip-1] = 2000.; //storing S2 data    
    } else {
      data_S0[i][j][k][iy-1][strip-1] = dataframe.sample(0).adc(); //storing S0 data
      data_S1[i][j][k][iy-1][strip-1] = dataframe.sample(1).adc(); //storing S1 data
      data_S2[i][j][k][iy-1][strip-1] = dataframe.sample(2).adc(); //storing S2 data    
    }
  }

  // Do CM 
  double sensor_data_S0[32];
  double sensor_data_S1[32];
  double sensor_data_S2[32];
  double CM1;
  double CM2;  
  int pedestal;
  if (doCM_) {
    for (int i=0; i<2; ++i) { 
      for (int j=0; j<6; ++j) {
	for (int k=0; k<4; ++k) {        
	  for (int m=0; m<5; ++m) {
	    
	    if (detType_ == 1 && k>1) continue;
	    
	    for (int n=0; n<32; ++n){
	      
	      pedestal = (int) hist_[i][j][k][m]->GetBinContent(n+1);
	      
	      sensor_data_S0[n] = data_S0[i][j][k][m][n] - pedestal;  //Read sensor data
	      sensor_data_S1[n] = data_S1[i][j][k][m][n] - pedestal;  //Read sensor data
	      sensor_data_S2[n] = data_S2[i][j][k][m][n] - pedestal;  //Read sensor data
	    } //n
	    
	    DoCommonMode(sensor_data_S0, &CM1, &CM2);  //Common mode calculation	  
	    CMN_S0[i][j][k][m][0] = CM1;
	    CMN_S0[i][j][k][m][1] = CM2;
	    
	    DoCommonMode(sensor_data_S1, &CM1, &CM2);  //Common mode calculation
	    CMN_S1[i][j][k][m][0] = CM1;
	    CMN_S1[i][j][k][m][1] = CM2;	  
	    
	    DoCommonMode(sensor_data_S2, &CM1, &CM2);  //Common mode calculation
	    CMN_S2[i][j][k][m][0] = CM1;
	    CMN_S2[i][j][k][m][1] = CM2;
	    
	  } //m
	} // k
      } // j
    } // i
  } // doCM

  // Dump RecHits
  for (idigi=digi->begin(); idigi!=digi->end(); ++idigi) {    
    
    dataframe = (*idigi);
    id = dataframe.id();
    plane = id.plane();
    ix    = id.six();
    iy    = id.siy();
    strip = id.strip();

    if (detType_ == 1) {
      j = (ix-1)/2;
      if (j<=5) i = 0;
      else i = 1;
      if (j>5) j=j-6;
      k = (ix-1)%2;
    } else if (detType_ == 2) {
      i = 0;
      j = plane-1;
      k = ix-1; 
    }
    
    if (ix==7 && iy==5 && strip==4) continue;

    pedestal = (int) hist_[i][j][k][iy-1]->GetBinContent(strip);
    /*
    if (strip<=16) 
      cout<<i+1<<" "<<j+1<<" "<<k+1<<" "<<iy<<" "<<strip<<" "<<pedestal<<" "<<CMN_S0[i][j][k][iy-1][0]<<" "<<CMN_S1[i][j][k][iy-1][0]<<" "<<CMN_S2[i][j][k][iy-1][0]<<endl;
    else
      cout<<i+1<<" "<<j+1<<" "<<k+1<<" "<<iy<<" "<<strip<<" "<<pedestal<<" "<<CMN_S0[i][j][k][iy-1][1]<<" "<<CMN_S1[i][j][k][iy-1][1]<<" "<<CMN_S2[i][j][k][iy-1][1]<<endl;    
    */
    if (tdc[7]>790) {
      if (strip<=16) 
	rec->push_back(algo_->reconstruct(*idigi, tdc[7], pedestal, CMN_S0[i][j][k][iy-1][0], CMN_S1[i][j][k][iy-1][0], CMN_S2[i][j][k][iy-1][0]));
      else 
	rec->push_back(algo_->reconstruct(*idigi, tdc[7], pedestal, CMN_S0[i][j][k][iy-1][1], CMN_S1[i][j][k][iy-1][1], CMN_S2[i][j][k][iy-1][1]));
    }

  }
  
  e.put(rec,rechitCollection_);
}



#include "EventFilter/ESRawToDigi/interface/ESRecHitProducerTB.h"
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
#include "TBDataFormats/HcalTBObjects/interface/HcalTBTriggerData.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBTiming.h"

ESRecHitProducerTB::ESRecHitProducerTB(ParameterSet const& ps)
{
  digiCollection_ = ps.getParameter<edm::InputTag>("ESdigiCollection");
  rechitCollection_ = ps.getParameter<std::string>("ESrechitCollection");
  produces<ESRecHitCollection>(rechitCollection_);
  
  gain_ = ps.getUntrackedParameter<int>("ESGain", 1);
  double ESMIPADC = ps.getUntrackedParameter<double>("ESMIPADC", 9);
  double ESMIPkeV = ps.getUntrackedParameter<double>("ESMIPkeV", 81.08);

  pedestalFile_ = ps.getUntrackedParameter<string>("PedestalFile");
  doCM_         = ps.getUntrackedParameter<bool>("doCM", true);
  sigma_        = ps.getUntrackedParameter<double>("SigmaForCM", 4);

  ped_ = new TFile(pedestalFile_.c_str());  //Root file with ped histos
  Char_t tmp[300];
  
  for (int i=0; i<2; ++i) { 
    for (int j=0; j<4; ++j) { 
      for (int k=0; k<4; ++k) {  
	sprintf(tmp, "DQMData/ES/QT/PedestalTB/ES Pedestal Fit Mean RMS Z 1 P %1d Col %02d Row %02d", i+1, j+1, k+1);
	hist_[i][j][k] = (TH1F*) ped_->Get(tmp);	    
      }          
    }
  }

  algo_ = new ESRecHitSimAlgoTB(gain_, ESMIPADC, ESMIPkeV); 
}

ESRecHitProducerTB::~ESRecHitProducerTB() {
  delete algo_;
  delete ped_;
}

void ESRecHitProducerTB::DoCommonMode(double det_data[], double *cm) {

  int current_min1 = 4095;
  int current_min2 = 4095;
  int current_strip_val1 = 0;
  int index = -1;
  int current_sum1 = 0;
  int threshold1=0;
  int n1=0;
  int check_bit1_10;
  int corrected_sum1;

  int treshold_const = (int) (4.3*sigma_);

  int myltip_factor[]={2048, 1024, 683, 512, 410, 341, 293, 256, 228, 205, 186, 171, 158, 146, 137, 128,
                       120, 114, 108, 102, 98, 93, 89, 85, 82, 79, 76, 73, 71, 68, 66, 64};

  for(int i=0; i<32; ++i) {
    current_strip_val1 = (int) det_data[i];
    if(current_min1 > current_strip_val1 ) {
      current_min1 = current_strip_val1;
      index = i;
    }
  }

  for(int i=0; i<32; ++i) {
    if (index==i) continue;
    current_strip_val1 = (int) det_data[i];
    if(current_min2 > current_strip_val1 ) {
      current_min2 = current_strip_val1;
    }
  }
  threshold1 = current_min2 + treshold_const;

  for(int i=0; i<32; ++i) {
    if (index==i) continue;
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

  *cm = (float)corrected_sum1;

}

void ESRecHitProducerTB::produce(Event& e, const EventSetup& es) {
  
  // Trigger and timing information
  Handle<HcalTBTriggerData> trg;
  Handle<HcalTBTiming> time;
  try {
    e.getByType(trg);
    e.getByType(time);
  }
  catch ( cms::Exception &e ) {
    LogDebug("") << "ESPedestal : Error! can't get trigger and/or timing information !" << std::endl;
  }

  int trgbit = 0;
  if( trg->wasBeamTrigger() )             trgbit = 1;
  if( trg->wasInSpillPedestalTrigger() )  trgbit = 2;
  if( trg->wasOutSpillPedestalTrigger() ) trgbit = 3;
  if( trg->wasLEDTrigger() )              trgbit = 4;
  if( trg->wasLaserTrigger() )            trgbit = 5;

  double t0 = 31;
  if (time->ttcL1Atime()>0 && time->BeamCoincidenceCount()>0) t0 =  time->ttcL1Atime() - time->BeamCoincidenceHits(0) - 1000;

  // Digis
  Handle<ESDigiCollection> digiHandle;  
  const ESDigiCollection* digi = 0;
  e.getByLabel( digiCollection_, digiHandle);
  digi = digiHandle.product();
  
  LogInfo("ESRecHitInfo") << "total # ESdigis: " << digi->size() ;  
  auto_ptr<ESRecHitCollection> rec(new EcalRecHitCollection());
    
  ESDigiCollection::const_iterator idigi;

  ESDataFrame dataframe;
  ESDetId id;
  
  //Need for storing original data 
  int data_S0[2][4][4][32];
  int data_S1[2][4][4][32];
  int data_S2[2][4][4][32];
  double CMN_S0[2][4][4];
  double CMN_S1[2][4][4];
  double CMN_S2[2][4][4];
  for (int i=0; i<2; ++i) 
    for (int j=0; j<4; ++j)
      for (int k=0; k<4; ++k) {
	CMN_S0[i][j][k] = 0;
	CMN_S1[i][j][k] = 0;
	CMN_S2[i][j][k] = 0;       
	for (int m=0; m<32; ++m) {
	  data_S0[i][j][k][m] = 0;
	  data_S1[i][j][k][m] = 0;
	  data_S2[i][j][k][m] = 0;
	}
      }

  // Store data for CM calculation
  int plane, ix, iy, strip;
  for (idigi=digi->begin(); idigi!=digi->end(); ++idigi) { 
    
    dataframe = (*idigi);
    id = dataframe.id();

    plane = id.plane();
    ix    = id.six();
    iy    = id.siy();
    strip = id.strip();    
    
    data_S0[plane-1][ix-1][iy-1][strip-1]=dataframe.sample(0).adc();
    data_S1[plane-1][ix-1][iy-1][strip-1]=dataframe.sample(1).adc();
    data_S2[plane-1][ix-1][iy-1][strip-1]=dataframe.sample(2).adc();
  }

  // Do CM 
  double sensor_data_S0[32];
  double sensor_data_S1[32];
  double sensor_data_S2[32];
  double cm;
  int pedestal;
  if (doCM_) {
    for (int i=0; i<2; ++i) { 
      for (int j=0; j<4; ++j) {
	for (int k=0; k<4; ++k) {        
	  for (int m=0; m<32; ++m) {
	    
	    // missing sensors
	    if (i==0 && j==0 && k==3) continue;
	    if (i==0 && j==3 && k==3) continue;	    

	    pedestal = (int) hist_[i][j][k]->GetBinContent(m+1);

	    sensor_data_S0[m] = data_S0[i][j][k][m] - pedestal;  //Read sensor data
	    sensor_data_S1[m] = data_S1[i][j][k][m] - pedestal;  //Read sensor data
	    sensor_data_S2[m] = data_S2[i][j][k][m] - pedestal;  //Read sensor data

	    // dead channel
	    if (i==1 && j==0 && k==3 && m==3) {
	      sensor_data_S0[m] = 4095;
	      sensor_data_S1[m] = 4095;
	      sensor_data_S2[m] = 4095;
	    }	    
	  } // m	    	 

	  DoCommonMode(sensor_data_S0, &cm);
	  CMN_S0[i][j][k] = cm;
	  
	  DoCommonMode(sensor_data_S1, &cm);
	  CMN_S1[i][j][k] = cm;
	  
	  DoCommonMode(sensor_data_S2, &cm);
	  CMN_S2[i][j][k] = cm;

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
    
    pedestal = (int) hist_[plane-1][ix-1][iy-1]->GetBinContent(strip);
    
    rec->push_back(algo_->reconstruct(*idigi, t0, pedestal, CMN_S0[plane-1][ix-1][iy-1], CMN_S1[plane-1][ix-1][iy-1], CMN_S2[plane-1][ix-1][iy-1]));
    
  }

  e.put(rec, rechitCollection_);
}



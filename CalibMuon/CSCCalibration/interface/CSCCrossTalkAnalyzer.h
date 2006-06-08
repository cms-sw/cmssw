/** 
 * Analyzer for calculating CFEB cross-talk & pedestal.
 * author S.Durkin, O.Boeriu 15/05/06 
 * runs over multiple DDUs
 * takes variable size chambers & layers  
 */

#include <iostream>
#include <time.h>
#include <sys/stat.h>	
#include <unistd.h>
#include <fstream>

#include "CalibMuon/CSCCalibration/interface/condbon.h"
#include "CalibMuon/CSCCalibration/interface/cscmap.h"
#include "CalibMuon/CSCCalibration/interface/CSCxTalk.h"
#include "TFile.h"
#include "TTree.h"
#include "TH1F.h"
#include "TH2F.h"

class TCalibCrossTalkEvt {
  public:
  Float_t xtalk_slope_left;
  Float_t xtalk_slope_right;
  Float_t xtalk_int_left;
  Float_t xtalk_int_right;
  Float_t xtalk_chi2_left;
  Float_t xtalk_chi2_right;
  Float_t peakTime;
  Int_t strip;
  Int_t layer;
  Int_t cham;
  Int_t ddu;
  Float_t pedMean;
  Float_t pedRMS;
  Float_t peakRMS;
  Float_t maxADC;
  Float_t sum;
  Float_t time[8];
  Float_t adc[8];
  Int_t tbin[8];
  Int_t event;
};

class CSCCrossTalkAnalyzer : public edm::EDAnalyzer {
 public:
  explicit CSCCrossTalkAnalyzer(edm::ParameterSet const& conf);
  virtual void analyze(edm::Event const& e, edm::EventSetup const& iSetup);
   
#define CHAMBERS_xt 18
#define LAYERS_xt 6
#define STRIPS_xt 80
#define TIMEBINS_xt 8
#define DDU_xt 36
#define TOTALSTRIPS_xt 480
#define TOTALEVENTS_xt 320

  ~CSCCrossTalkAnalyzer(){

   Conv binsConv;
    //get time of Run file for DB transfer
    filein.open("../test/CSCxtalk.cfg");
    filein.ignore(1000,'\n');
    
    while(filein != NULL){
      lines++;
      getline(filein,PSet);
      
      if (lines==3){
	name=PSet;  
      }
    }
    
    string::size_type runNameStart = name.find("RunNum",0);
    string::size_type runNameEnd   = name.find("bin",0);
    int nameSize = runNameEnd+3-runNameStart;
    std::string myname= name.substr(runNameStart,nameSize);
    
    struct tm* clock;			    
    struct stat attrib;			    
    stat(myname.c_str(), &attrib);          
    clock = localtime(&(attrib.st_mtime));  
    std::string myTime=asctime(clock);

    //DB object and map
    CSCobject *cn = new CSCobject();
    CSCobject *cn1 = new CSCobject();
    cscmap *map = new cscmap();
    condbon *dbon = new condbon();

    //root ntuple
   TCalibCrossTalkEvt calib_evt;
   TFile calibfile("ntuples/calibxtalk.root", "RECREATE");
   TTree calibtree("Calibration","Cross-talk");
   calibtree.Branch("EVENT", &calib_evt, "xtalk_slope_left/F:xtalk_slope_right/F:xtalk_int_left/F:xtalk_int_right/F:xtalk_chi2_left/F:xtalk_chi2_right/F:peakTime/F:strip/I:layer/I:cham/I:ddu/I:pedMean/F:pedRMS/F:peakRMS/F:maxADC/F:sum/F:time[8]/F:adc[8]/F:tbin[8]/I:event/I");
   
    ////////////////////////////////////////////////////////////////////iuse==strip-1
    // Now that we have filled our array, extract convd and nconvd
    float adc_ped_sub_left = -999.;
    float adc_ped_sub = -999.;
    float adc_ped_sub_right = -999.;
    int thebin;
    float sum=0.0;
    float mean=0;
    
    for (int iii=0; iii<Nddu; iii++){
      calib_evt.event=myevt;
      for (int i=0; i<NChambers; i++){

	//get chamber ID from DB mapping        
	int new_crateID = crateID[i];
	int new_dmbID   = dmbID[i];
	std::cout<<" Crate: "<<new_crateID<<" and DMB:  "<<new_dmbID<<std::endl;
	map->crate_chamber(new_crateID,new_dmbID,&chamber_id,&chamber_num,&sector);
	std::cout<<"Data is for chamber:: "<< chamber_id<<" in sector:  "<<sector<<std::endl;

	meanPedestal = 0.0;
	meanPeak     = 0.0;
	meanPeakSquare=0.0;
	meanPedestalSquare = 0.;
	theRMS      =0.0;
	thePedestal =0.0;
	theRSquare  =0.0;
	thePeak     =0.0;
	thePeakRMS  =0.0;
	theSumFive  =0.0;

	for (int j=0; j<LAYERS_xt; j++){
	  mean=0.,sum=0.;
	  for (int s=0; s<size[i]; s++) {
	    //re-zero convd and nconvd
	    for (int m=0; m<3; m++){
	      for (int n=0; n<120; n++){
		binsConv.convd[m][n]  = 0.;
		binsConv.nconvd[m][n] = 0.;
	      }
	    }
	    
	    for (int l=0; l < TIMEBINS_xt*20; l++){
	      adc_ped_sub_left  = theadccountsl[iii][i][j][s][l] - (theadccountsl[iii][i][j][s][0] + theadccountsl[iii][i][j][s][1])/2.;
	      adc_ped_sub       = theadccountsc[iii][i][j][s][l] - (theadccountsc[iii][i][j][s][0] + theadccountsc[iii][i][j][s][1])/2.;
	      adc_ped_sub_right = theadccountsr[iii][i][j][s][l] - (theadccountsr[iii][i][j][s][0] + theadccountsr[iii][i][j][s][1])/2.;
	      
	      thebin=thebins[iii][i][j][s][l];
	    
	      if (thebin >= 0 && thebin < 120){
		binsConv.convd[0][thebin]  += adc_ped_sub_left;
		binsConv.nconvd[0][thebin] += 1.0;
		
		binsConv.convd[1][thebin]  += adc_ped_sub;
		binsConv.nconvd[1][thebin] += 1.0;
		
		binsConv.convd[2][thebin]  += adc_ped_sub_right;
		binsConv.nconvd[2][thebin] += 1.0;
		
	      }
	    } //loop over timebins
	    
	    for (int m=0; m<3; m++){
	      for (int n=0; n<120; n++){
		if(binsConv.nconvd[m][n]>1.0 && binsConv.nconvd[m][n] !=0.){
		  binsConv.convd[m][n] = binsConv.convd[m][n]/binsConv.nconvd[m][n];
		}
	      }
	    }
	    
	    // Call our functions first time to calculate mean pf peak time over a layer
	    float xl_temp_a = 0.0;
	    float xl_temp_b = 0.0;
	    float minl_temp = 0.0;
	    float xr_temp_a = 0.0;
	    float xr_temp_b = 0.0;
	    float minr_temp = 0.0;
	    float pTime     = 0.0;
	    
	    binsConv.mkbins(50.);
	    binsConv.convolution(&xl_temp_a, &xl_temp_b, &minl_temp, &xr_temp_a, &xr_temp_b, &minr_temp, &pTime);
	    myPeakTime[iii][i][j][s] = pTime;
	    sum=sum+myPeakTime[iii][i][j][s];
	    mean = sum/size[i];
	  }
	  
	  int layer_id=chamber_num+j+1;
	  if(sector==-100)continue;
	  cn->obj[layer_id].resize(size[i]);
	  cn1->obj[layer_id].resize(size[i]);
	  for (int k=0; k<size[i]; k++){
	    // re-zero convd and nconvd 
	    for (int m=0; m<3; m++){
	      for (int n=0; n<120; n++){
		binsConv.convd[m][n]  = 0.;
		binsConv.nconvd[m][n] = 0.;
	      }
	    }
	    
	    for (int l=0; l < TIMEBINS_xt*20; l++){
	      adc_ped_sub_left  = theadccountsl[iii][i][j][k][l] - (theadccountsl[iii][i][j][k][0] + theadccountsl[iii][i][j][k][1])/2.;	  
	      adc_ped_sub       = theadccountsc[iii][i][j][k][l] - (theadccountsc[iii][i][j][k][0] + theadccountsc[iii][i][j][k][1])/2.;
	      adc_ped_sub_right = theadccountsr[iii][i][j][k][l] - (theadccountsr[iii][i][j][k][0] + theadccountsr[iii][i][j][k][1])/2.;
	      
	      thebin=thebins[iii][i][j][k][l];
	      
	      if (thebin >= 0 && thebin < 120){
		binsConv.convd[0][thebin]  += adc_ped_sub_left;
		binsConv.nconvd[0][thebin] += 1.0;
		
		binsConv.convd[1][thebin]  += adc_ped_sub;
		binsConv.nconvd[1][thebin] += 1.0;
		
		binsConv.convd[2][thebin]  += adc_ped_sub_right;
		binsConv.nconvd[2][thebin] += 1.0;
		
	      }
	    } //loop over timebins
	    
	    for (int m=0; m<3; m++){
	      for (int n=0; n<120; n++){
		if(binsConv.nconvd[m][n]>1.0 && binsConv.nconvd[m][n] !=0.){
		  binsConv.convd[m][n] = binsConv.convd[m][n]/binsConv.nconvd[m][n];
		}
	      }
	    }
	    //////////////////////////////////////////////////////////////////
	    // Call our functions a second time to calculate the cross-talk //
	    //////////////////////////////////////////////////////////////////
	    float xl_temp_a = 0.;
	    float xl_temp_b = 0.;
	    float minl_temp = 0.;
	    float xr_temp_a = 0.;
	    float xr_temp_b = 0.;
	    float minr_temp = 0.;
	    float pTime     = 0.;
	    
	    binsConv.mkbins(50.);
	    binsConv.convolution(&xl_temp_a, &xl_temp_b, &minl_temp, &xr_temp_a, &xr_temp_b, &minr_temp, &pTime);
	    
	    if (k==0){
	      xtalk_intercept_left[iii][i][j][k]  = 0.0;
	      xtalk_slope_left[iii][i][j][k]      = 0.0;
	      xtalk_chi2_left[iii][i][j][k]       = 0.0;
	      //right side is calculated
	      xtalk_slope_right[iii][i][j][k]     = xl_temp_b;
	      xtalk_intercept_right[iii][i][j][k] = xl_temp_a;
	      xtalk_chi2_right[iii][i][j][k]      = minl_temp;
	      myPeakTime[iii][i][j][k]            = pTime;
	    }else if(k==size[i]-1){
	      xtalk_intercept_right[iii][i][j][k]  = 0.0;
	      xtalk_slope_right[iii][i][j][k]      = 0.0;
	      xtalk_chi2_right[iii][i][j][k]       = 0.0;
	      //left side is calculated
	      xtalk_intercept_left[iii][i][j][k]   = xr_temp_a;
	      xtalk_slope_left[iii][i][j][k]       = xr_temp_b;
	      xtalk_chi2_left[iii][i][j][k]        = minr_temp;
	      myPeakTime[iii][i][j][k]             = pTime;
	    }else{
	      xtalk_intercept_left[iii][i][j][k]  = xl_temp_a;
	      xtalk_intercept_right[iii][i][j][k] = xr_temp_a;
	      xtalk_slope_left[iii][i][j][k]      = xl_temp_b;
	      xtalk_slope_right[iii][i][j][k]     = xr_temp_b;
	      xtalk_chi2_left[iii][i][j][k]       = minl_temp;
	      xtalk_chi2_right[iii][i][j][k]      = minr_temp;
	      myPeakTime[iii][i][j][k]            = pTime;
	    }
	    
	    fff = (j*size[i])+k;
	    float the_xtalk_left_a  = xtalk_intercept_left[iii][i][j][k];
	    float the_xtalk_right_a = xtalk_intercept_right[iii][i][j][k];
	    float the_xtalk_left_b  = xtalk_slope_left[iii][i][j][k];
	    float the_xtalk_right_b = xtalk_slope_right[iii][i][j][k];
	    float the_chi2_right    = xtalk_chi2_right[iii][i][j][k];
	    float the_chi2_left     = xtalk_chi2_left[iii][i][j][k];
	    float the_peakTime      = myPeakTime[iii][i][j][k]; 
	  
	    new_xtalk_intercept_right[fff] = the_xtalk_right_a ;
	    new_xtalk_intercept_left[fff]  = the_xtalk_left_a ;
	    new_xtalk_slope_right[fff]     = the_xtalk_right_b ;
	    new_xtalk_slope_left[fff]      = the_xtalk_left_b ;
	    new_rchi2[fff]                 = the_chi2_right;
	    new_lchi2[fff]                 = the_chi2_left;
	    newPeakTime[fff]               = the_peakTime;
	    newMeanPeakTime[fff]           = the_peakTime-mean;

	    //pedestal info
	    thePedestal  = arrayPed[iii][i][j][k];
	    meanPedestal = arrayOfPed[iii][i][j][k]/evt;
	    newPed[fff]  = meanPedestal;
	    meanPedestalSquare = arrayOfPedSquare[iii][i][j][k] / evt;
	    theRMS       = sqrt(abs(meanPedestalSquare - meanPedestal*meanPedestal));
	    newRMS[fff]  = theRMS;
	    theRSquare   = (thePedestal-meanPedestal)*(thePedestal-meanPedestal)/(theRMS*theRMS*theRMS*theRMS);
	    thePeak      = arrayPeak[iii][i][j][k];
	    meanPeak     = arrayOfPeak[iii][i][j][k] / evt;
	    meanPeakSquare = arrayOfPeakSquare[iii][i][j][k] / evt;
	    thePeakRMS   = sqrt(abs(meanPeakSquare - meanPeak*meanPeak));
	    newPeakRMS[fff] = thePeakRMS;
	    newPeak[fff] = thePeak;
	    
	    theSumFive = arraySumFive[iii][i][j][k];
	    newSumFive[fff]=theSumFive;
	    
	    calib_evt.pedMean  = newPed[fff];
	    calib_evt.pedRMS   = newRMS[fff];
	    calib_evt.peakRMS  = newPeakRMS[fff];
	    calib_evt.maxADC   = newPeak[fff];
	    calib_evt.sum      = newSumFive[fff];
	    
	    std::cout <<"Ch "<<i<<" L "<<j<<" S "<<k<<"  ped "<<meanPedestal<<" RMS "<<theRMS<<" maxADC "<<thePeak<<" maxRMS "<<thePeakRMS<<" Sum/peak "<<theSumFive<<" IntL "<<the_xtalk_left_a<<" SL "<<the_xtalk_left_b<<" IntR "<<the_xtalk_right_a<<" SR "<<the_xtalk_right_b<<" diff "<<the_peakTime-mean<<std::endl;
	    calib_evt.xtalk_slope_left     = xtalk_slope_left[iii][i][j][k];
	    calib_evt.xtalk_slope_right    = xtalk_slope_right[iii][i][j][k];
	    calib_evt.xtalk_int_left       = xtalk_intercept_left[iii][i][j][k];
	    calib_evt.xtalk_int_right      = xtalk_intercept_right[iii][i][j][k];
	    calib_evt.xtalk_chi2_left      = xtalk_chi2_left[iii][i][j][k];
	    calib_evt.xtalk_chi2_right     = xtalk_chi2_right[iii][i][j][k];
	    calib_evt.peakTime             = myPeakTime[iii][i][j][k];
	    calib_evt.cham                 = i;
	    calib_evt.ddu                  = iii;
	    calib_evt.layer                = j;
	    calib_evt.strip                = k;
	   

	    /* for (int tbin=0;tbin<8;tbin++){ */
/* 	      calib_evt.time[tbin] = myTime[tbin]; */
/* 	      calib_evt.adc[tbin]  = myADC[tbin]; */
/* 	      calib_evt.tbin[tbin] = myTbin[tbin]; */
/*  	    } */
	    
	    calibtree.Fill();
	    cn->obj[layer_id][k].resize(2);
	    cn->obj[layer_id][k][0] = meanPedestal;
	    cn->obj[layer_id][k][1] = theRMS;
	    cn1->obj[layer_id][k].resize(6);
	    cn1->obj[layer_id][k][0] = the_xtalk_right_b ;
	    cn1->obj[layer_id][k][1] = the_xtalk_right_a ;
	    cn1->obj[layer_id][k][2] = the_chi2_right;
	    cn1->obj[layer_id][k][3] = the_xtalk_left_b ;
	    cn1->obj[layer_id][k][4] = the_xtalk_left_a ;
	    cn1->obj[layer_id][k][5] = the_chi2_left;

	  }//loop over strips
	}//loop over layers
      }//chambers
    }//Nddu
   
    dbon->cdbon_last_run("pedestals",&run);
    std::cout<<"Last pedestal run "<<run<<std::endl;
    if(debug) dbon->cdbon_write(cn,"pedestals",run+1,myTime);
    dbon->cdbon_last_run("crosstalk",&run);
    std::cout<<"Last crosstalk run "<<run<<" for run file "<<myname<<" saved "<<myTime<<std::endl;
    if(debug) dbon->cdbon_write(cn1,"crosstalk",run+1,myTime);
    calibfile.Write();    
    calibfile.Close();  
  }
  
 private:
  int eventNumber,evt,strip,misMatch,fff,ret_code,length,Nddu,myevt;
  int i_chamber,i_layer,reportedChambers,chamber_num,sector,run,NChambers ;
  int dmbID[CHAMBERS_xt],crateID[CHAMBERS_xt],size[CHAMBERS_xt];
  std::vector<int> adc;
  std::string chamber_id;
  std::string tid;
  int thebins[DDU_xt][CHAMBERS_xt][LAYERS_xt][STRIPS_xt][TIMEBINS_xt*20];
  int theadccountsc[DDU_xt][CHAMBERS_xt][LAYERS_xt][STRIPS_xt][TIMEBINS_xt*20];
  int theadccountsl[DDU_xt][CHAMBERS_xt][LAYERS_xt][STRIPS_xt][TIMEBINS_xt*20];
  int theadccountsr[DDU_xt][CHAMBERS_xt][LAYERS_xt][STRIPS_xt][TIMEBINS_xt*20];
  float pedMean,pedMean1,time,max1,max2,aPeak,sumFive;
  float meanPedestal,meanPeak,meanPeakSquare,meanPedestalSquare,theRMS;
  float thePeak,thePeakRMS,theSumFive,thePedestal,theRSquare;
  float thetime[DDU_xt][CHAMBERS_xt][LAYERS_xt][STRIPS_xt][TIMEBINS_xt*20];
  float xtalk_intercept_left[DDU_xt][CHAMBERS_xt][LAYERS_xt][STRIPS_xt];
  float xtalk_intercept_right[DDU_xt][CHAMBERS_xt][LAYERS_xt][STRIPS_xt];
  float xtalk_slope_left[DDU_xt][CHAMBERS_xt][LAYERS_xt][STRIPS_xt];
  float xtalk_slope_right[DDU_xt][CHAMBERS_xt][LAYERS_xt][STRIPS_xt];
  float xtalk_chi2_left[DDU_xt][CHAMBERS_xt][LAYERS_xt][STRIPS_xt];
  float xtalk_chi2_right[DDU_xt][CHAMBERS_xt][LAYERS_xt][STRIPS_xt];
  float myPeakTime[DDU_xt][CHAMBERS_xt][LAYERS_xt][STRIPS_xt];
  float myMeanPeakTime[DDU_xt][CHAMBERS_xt][LAYERS_xt][STRIPS_xt];
  float array_meanPeakTime[DDU_xt][CHAMBERS_xt][LAYERS_xt][STRIPS_xt];
  float arrayOfPed[DDU_xt][CHAMBERS_xt][LAYERS_xt][STRIPS_xt];
  float arrayOfPedSquare[DDU_xt][CHAMBERS_xt][LAYERS_xt][STRIPS_xt];
  float arrayPed[DDU_xt][CHAMBERS_xt][LAYERS_xt][STRIPS_xt];
  float arrayPeak[DDU_xt][CHAMBERS_xt][LAYERS_xt][STRIPS_xt];
  float arrayOfPeak[DDU_xt][CHAMBERS_xt][LAYERS_xt][STRIPS_xt];
  float arrayOfPeakSquare[DDU_xt][CHAMBERS_xt][LAYERS_xt][STRIPS_xt];
  float arraySumFive[DDU_xt][CHAMBERS_xt][LAYERS_xt][STRIPS_xt];
  float myTime[TIMEBINS_xt];
  float myADC[TIMEBINS_xt];
  int myTbin[TIMEBINS_xt];
  float newPed[TOTALSTRIPS_xt];
  float newRMS[TOTALSTRIPS_xt];
  float newPeakRMS[TOTALSTRIPS_xt];
  float newPeak[TOTALSTRIPS_xt];
  float newSumFive[TOTALSTRIPS_xt];
  float new_xtalk_intercept_right[TOTALSTRIPS_xt];
  float new_xtalk_intercept_left[TOTALSTRIPS_xt];
  float new_xtalk_slope_right[TOTALSTRIPS_xt];
  float new_xtalk_slope_left[TOTALSTRIPS_xt];
  float new_rchi2[TOTALSTRIPS_xt];
  float new_lchi2[TOTALSTRIPS_xt];
  float newPeakTime[TOTALSTRIPS_xt];
  float newMeanPeakTime[TOTALSTRIPS_xt];
  int lines;
  std::ifstream filein;
  string PSet,name;
  bool debug;

  //root ntuple
  TCalibCrossTalkEvt calib_evt;
  TBranch *calibevt;
  TTree *calibtree;
  TFile *calibfile;

  /* calibfile = new TFile("ntuples/calibxtalk.root", "RECREATE","Calibration Ntuple"); */
/*   calibtree = new TTree("Calibration","Crosstalk"); */
/*   calibevt = calibtree->Branch("EVENT", &calib_evt, ""xtalk_slope_left/F:xtalk_slope_right/F:xtalk_int_left/F:xtalk_int_right/F:xtalk_chi2_left/F:xtalk_chi2_right/F:peakTime/F:strip/I:layer/I:cham/I:ddu/I:pedMean/F:pedRMS/F:peakRMS/F:maxADC/F:sum/F:time[8]/F:adc[8]/F:tbin[8]/I:event/I"); */
};


/** 
 * Analyzer for reading gains information
 * author S. Durkin, O.Boeriu 18/03/06 
 *   
 */

#include <iostream>
#include "CalibMuon/CSCCalibration/interface/condbc.h"
#include "CalibMuon/CSCCalibration/interface/cscmap.h"
#include "/afs/cern.ch/cms/external/lcg/external/root/5.08.00/slc3_ia32_gcc323/root/include/TFile.h"
#include "/afs/cern.ch/cms/external/lcg/external/root/5.08.00/slc3_ia32_gcc323/root/include/TTree.h"

class TCalibEvt {
  public:
  Float_t slope;
  Float_t intercept;
  Float_t chi2;
  Int_t strip;
  Int_t layer;
  Int_t cham;
};

class CSCGainAnalyzer : public edm::EDAnalyzer {
 public:
  explicit CSCGainAnalyzer(edm::ParameterSet const& conf);
  virtual void analyze(edm::Event const& e, edm::EventSetup const& iSetup);
  
#define CHAMBERS 9
#define LAYERS 6
#define STRIPS 80
#define NUMBERPLOTTED 10 
#define NUMMODTEN 200
#define DDU 2

  ~CSCGainAnalyzer(){
   
   //create array for database transfer
   condbc *cdb = new condbc();
   cscmap *map = new cscmap();
   
   //root ntuple

   TCalibEvt calib_evt;
   TFile calibfile("ntuples/calibgain.root", "RECREATE");
   TTree calibtree("Calibration","Gains");
   calibtree.Branch("EVENT", &calib_evt, "slope/F:intercept/F:chi2/F:strip/I:layer/I:cham/I");
   
   //create array for database transfer
   
   for (int dduiter=0;dduiter<Nddu;dduiter++){
     for(int chamberiter=0; chamberiter<NChambers; chamberiter++){
     
       float the_gain_sq   = 0.;
       float the_gain      = 0.;
       float the_intercept = 0.; 
       float the_chi2      = 0.;
       
       for (int cham=0;cham<NChambers;cham++){
	 if (cham !=chamberiter) continue;
	 
	 for (int layeriter=0; layeriter<LAYERS; layeriter++){
	   for (int stripiter=0; stripiter<STRIPS; stripiter++){
	     
	     for (int j=0; j<LAYERS; j++){//layer
	       if (j != layeriter) continue;
	       
	       for (int k=0; k<size[cham]; k++){//strip
		 if (k != stripiter) continue;
		 float sumOfX    = 0.0;
		 float sumOfY    = 0.0;
		 float sumOfXY   = 0.0;
		 float sumx2     = 0.0;
		 float gainSlope = 0.0;
		 float gainIntercept = 0.0;
		 float chi2      = 0.0;
		 
		 float charge[NUMBERPLOTTED]={22.4, 44.8, 67.2, 89.6, 112, 134.4, 156.8, 179.2, 201.6, 224.0};// 246.4, 268.8, 291.2, 313.6, 336.0, 358.4, 380.8, 403.2, 425.6, 448}
		 
		 for(int ii=0; ii<NUMBERPLOTTED; ii++){//numbers    
		   sumOfX  += charge[ii];
		   sumOfY  += maxmodten[ii][cham][j][k];
		   sumOfXY += (charge[ii]*maxmodten[ii][cham][j][k]);
		   sumx2   += (charge[ii]*charge[ii]);
		   //std::cout <<"Maxmodten "<<maxmodten[ii][cham][j][k]<<std::endl;
		 }
		 
		 //Fit parameters
		 gainSlope     = ((NUMBERPLOTTED*sumOfXY) - (sumOfX * sumOfY))/((NUMBERPLOTTED*sumx2) - (sumOfX*sumOfX));//k
		 gainIntercept = ((sumOfY*sumx2)-(sumOfX*sumOfXY))/((NUMBERPLOTTED*sumx2)-(sumOfX*sumOfX));//m
		 
		 for(int ii=0; ii<NUMBERPLOTTED; ii++){
		   chi2  += (maxmodten[ii][cham][j][k]-(gainIntercept+(gainSlope*charge[ii])))*(maxmodten[ii][cham][j][k]-(gainIntercept+(gainSlope*charge[ii])))/(NUMBERPLOTTED*NUMBERPLOTTED);
		 }
		 
		 arrayOfGain[dduiter][cham][j][k]            = gainSlope;
		 arrayOfGainSquare[dduiter][cham][j][k]      = gainSlope*gainSlope;
		 arrayOfIntercept[dduiter][cham][j][k]       = gainIntercept;
		 arrayOfInterceptSquare[dduiter][cham][j][k] = gainIntercept*gainIntercept; 
		 arrayOfChi2[dduiter][cham][j][k]            = chi2;
		 
		 the_gain          = arrayOfGain[dduiter][cham][j][k];
		 the_gain_sq       = arrayOfGainSquare[dduiter][cham][j][k];
		 the_intercept     = arrayOfIntercept[dduiter][cham][j][k];
		 the_chi2          = arrayOfChi2[dduiter][cham][j][k];
		 
		 fff = (j*size[cham])+k; 
		 
		 newIntercept[fff] = the_intercept;
		 newGain[fff]      = the_gain;
		 newChi2[fff]      = the_chi2;
		 
		 std::cout <<"Chamber: "<<cham<<" Layer:   "<<j<<" Strip:   "<<fff<<"  Slope:    "<<newGain[fff] <<"    Intercept:    "<<newIntercept[fff] <<"        chi2 "<<newChi2[fff]<<std::endl;
		 
		 calib_evt.slope     = gainSlope;
		 calib_evt.intercept = gainIntercept;
		 calib_evt.chi2      = chi2;
		 calib_evt.strip     = k;
		 calib_evt.layer     = j;
		 calib_evt.cham      = cham;
		 
		 calibtree.Fill();
	       }//k loop
	     }//j loop
	   }//stripiter loop
	 }//layiter loop
	 //get chamber ID from Igor's mapping        
	 int new_crateID = crateID[cham];
	 int new_dmbID   = dmbID[cham];
	 std::cout<<" Crate: "<<new_crateID<<" and DMB:  "<<new_dmbID<<std::endl;
	 map->crate_chamber(new_crateID,new_dmbID,&chamber_id,&chamber_num,&sector);
	 std::cout<<" Above data is for chamber:: "<< chamber_id<<" and sector:  "<<sector<<std::endl;
	 //info needed for database
	 string test1="CSC_slice";
	 string test2="gain_slope";
	 string test3="gain_intercept";
	 string test4="gain_chi2";
	 std::string answer;
	 
	 std::cout<<" DO you want to send constants to DB? "<<std::endl;
	 std::cout<<" Please answer y or n for EACH chamber present! "<<std::endl;
	 
	 std::cin>>answer;
	 if(answer=="y"){
	   //SEND CONSTANTS TO DB
	   if (eventNumber !=3200){std::cout<<"Number of events not as expected!"<<eventNumber<<std::endl; continue;}
	   cdb->cdb_write(test1,chamber_id,chamber_num,test2,size[cham]*6, newGain,     6, &ret_code);
	   cdb->cdb_write(test1,chamber_id,chamber_num,test3,size[cham]*6, newIntercept,6, &ret_code);
	   cdb->cdb_write(test1,chamber_id,chamber_num,test4,size[cham]*6, newChi2,     6, &ret_code);
	   
	   std::cout<<" Data SENT to DB! "   <<std::endl;
	 }else{
	   std::cout<<" NO data was sent!!! "<<std::endl;
	 }
       }//cham 
     }//chamberiter
   }//dduiter
   calibfile.Write();
   calibfile.Close();
  }

 private:
  std::vector<int> newadc; 
  std::string chamber_id;
  int eventNumber,evt,chamber_num,sector,i_chamber,i_layer,reportedChambers;
  int fff,ret_code,length,strip,misMatch,NChambers,Nddu;
  int dmbID[CHAMBERS],crateID[CHAMBERS],size[CHAMBERS]; 
  float gainSlope,gainIntercept;
  float adcMax[DDU][CHAMBERS][LAYERS][STRIPS];
  float adcMean_max[DDU][CHAMBERS][LAYERS][STRIPS];
  float arrayOfGain[DDU][CHAMBERS][LAYERS][STRIPS];
  float arrayOfGainSquare[DDU][CHAMBERS][LAYERS][STRIPS];
  float arrayOfIntercept[DDU][CHAMBERS][LAYERS][STRIPS];
  float arrayOfChi2[DDU][CHAMBERS][LAYERS][STRIPS];
  float arrayOfInterceptSquare[DDU][CHAMBERS][LAYERS][STRIPS];
  float maxmodten[NUMMODTEN][CHAMBERS][LAYERS][STRIPS];
  float newGain[480];
  float newIntercept[480];
  float newChi2[480];
  
};

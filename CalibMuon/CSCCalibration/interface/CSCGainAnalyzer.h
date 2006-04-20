/** 
 * Analyzer for reading gains information
 * author O.Boeriu 18/03/06 
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
  Int_t strip;
  Int_t layer;
  Int_t cham;
};

class CSCGainAnalyzer : public edm::EDAnalyzer {
 public:
  explicit CSCGainAnalyzer(edm::ParameterSet const& conf);
  virtual void analyze(edm::Event const& e, edm::EventSetup const& iSetup);
  
#define CHAMBERS 5
#define LAYERS 6
#define STRIPS 80
#define NUMBERPLOTTED 10 
#define NUMMODTEN 200

 int evt;
 std::vector<int> newadc;
 int dmbID[CHAMBERS],crateID[CHAMBERS],chamber_num,sector;
 int i_chamber,i_layer,reportedChambers ;
 int fff,ret_code,length;
 std::string chamber_id;
 int strip,misMatch;
 float gainSlope,gainIntercept;
 
 //definition of arrays
 float adcMax[CHAMBERS][LAYERS][STRIPS];
 float adcMean_max[CHAMBERS][LAYERS][STRIPS];
 float arrayOfGain[CHAMBERS][LAYERS][STRIPS];
 float arrayOfGainSquare[CHAMBERS][LAYERS][STRIPS];
 float arrayOfIntercept[CHAMBERS][LAYERS][STRIPS];
 float arrayOfChi2[CHAMBERS][LAYERS][STRIPS];
 float arrayOfInterceptSquare[CHAMBERS][LAYERS][STRIPS];
 float newGain[480];
 float newIntercept[480];
 float newChi2[480];
 float maxmodten[NUMMODTEN][CHAMBERS][LAYERS][STRIPS];
 
 ~CSCGainAnalyzer(){
   
   //create array (480 entries) for database transfer
   condbc *cdb = new condbc();
   cscmap *map = new cscmap();
   
   //root ntuple

   TCalibEvt calib_evt;
   TFile calibfile("calibgain.root", "RECREATE");
   TTree calibtree("Calibration","Gain");
   calibtree.Branch("EVENT", &calib_evt, "slope/F:intercept/F:strip/I:layer/I:cham/I");
   
   //create array (480 entries) for database transfer
   for(int chamberiter=0; chamberiter<CHAMBERS; chamberiter++){
     
    float the_gain_sq   = 0.;
    float the_gain      = 0.;
    float the_intercept = 0.; 
    float the_chi2      = 0.;
      
     for (int cham=0;cham<CHAMBERS;cham++){
       if (cham !=chamberiter) continue;
       
       for (int layeriter=0; layeriter<LAYERS; layeriter++){
	 for (int stripiter=0; stripiter<STRIPS; stripiter++){
	   
	   for (int j=0; j<LAYERS; j++){//layer
	     if (j != layeriter) continue;
	     
	     for (int k=0; k<STRIPS; k++){//strip
	       if (k != stripiter) continue;
	      float sumOfX    = 0.0;
	      float sumOfY    = 0.0;
	      float sumOfXY   = 0.0;
	      float sumx2     = 0.0;
	      float gainSlope = 0.0;
	      float gainIntercept = 0.0;
	      float chi2      = 0.0;
		
	       float charge[10]={22.4, 44.8, 67.2, 89.6, 112, 134.4, 156.8, 179.2, 201.6, 224.0};
	       
	       for(int ii=0; ii<10; ii++){//numbers    
		 sumOfX  += charge[ii];
		 sumOfY  += maxmodten[ii][cham][j][k];
		 sumOfXY += (charge[ii]*maxmodten[ii][cham][j][k]);
		 sumx2   += (charge[ii]*charge[ii]);
	       }
	       
	       //Fit parameters
	       gainSlope     = ((10*sumOfXY) - (sumOfX * sumOfY))/((10*sumx2) - (sumOfX*sumOfX));//k
	       gainIntercept = ((sumOfY*sumx2)-(sumOfX*sumOfXY))/((10*sumx2)-(sumOfX*sumOfX));//m
	       
	       for(int ii=0; ii<10; ii++){
		 chi2  += (maxmodten[ii][cham][j][k]-(gainIntercept+(gainSlope*charge[ii])))*(maxmodten[ii][cham][j][k]-(gainIntercept+(gainSlope*charge[ii])))/100.;
	       }
	       
	       arrayOfGain[cham][j][k]            = gainSlope;
	       arrayOfGainSquare[cham][j][k]      = gainSlope*gainSlope;
	       arrayOfIntercept[cham][j][k]       = gainIntercept;
	       arrayOfInterceptSquare[cham][j][k] = gainIntercept*gainIntercept; 
	       arrayOfChi2[cham][j][k]            = chi2;
	       
	       fff = (j*80)+k; //this is for 480 entries in the array
	       
	       the_gain          = arrayOfGain[cham][j][k];
	       the_gain_sq       = arrayOfGainSquare[cham][j][k];
	       the_intercept     = arrayOfIntercept[cham][j][k];
	       the_chi2          = arrayOfChi2[cham][j][k];
	       
	       newIntercept[fff] = the_intercept;
	       newGain[fff]      = the_gain;
	       newChi2[fff]      = the_chi2;

	       std::cout <<"Chamber: "<<cham<<" Layer:   "<<j<<" Strip:   "<<fff<<"  Slope:    "<<newGain[fff] <<"    Intercept:    "<<newIntercept[fff] <<"        chi2 "<<newChi2[fff]<<std::endl;
	       
	       calib_evt.slope     = gainSlope;
	       calib_evt.intercept = gainIntercept;
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
	 cdb->cdb_write(test1,chamber_id,chamber_num,test2,480, newGain,     3, &ret_code);
	 cdb->cdb_write(test1,chamber_id,chamber_num,test3,480, newIntercept,3, &ret_code);
	 cdb->cdb_write(test1,chamber_id,chamber_num,test4,480, newChi2,     3, &ret_code);
	 
	 std::cout<<" Data SENT to DB! "   <<std::endl;
       }else{
	 std::cout<<" NO data was sent!!! "<<std::endl;
       }
     }//cham loop
   }//chamberiter loop
   calibfile.Write();
   calibfile.Close();
 }
    
 private:
 // variables persistent across events should be declared here.
 int eventNumber;
};
  

/** 
 * Analyzer for reading gains information
 * author S. Durkin, O.Boeriu 18/03/06 
 *   
 */

#include <iostream>
#include <time.h>
#include <sys/stat.h>	
#include <unistd.h>
#include <fstream>
#include "CalibMuon/CSCCalibration/interface/condbon.h"
#include "CalibMuon/CSCCalibration/interface/cscmap.h"
#include "TFile.h"
#include "TTree.h"

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
  
#define CHAMBERS 468
#define LAYERS 6
#define STRIPS 80
#define NUMBERPLOTTED 10 
#define NUMMODTEN 200
#define DDU 36

  ~CSCGainAnalyzer(){
//get time of Run file for DB transfer
    filein.open("../test/CSCgain.cfg");
    filein.ignore(1000,'\n');
    
    while(filein != NULL){
      lines++;
      getline(filein,PSet);
      
      if (lines==3){
	name=PSet;  
	cout<<name<<endl;
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
    cscmap *map = new cscmap();
    condbon *dbon = new condbon();
    
    //root ntuple information
    TCalibEvt calib_evt;
    TFile calibfile("ntuples/calibgain.root", "RECREATE");
    TTree calibtree("Calibration","Gains");
    calibtree.Branch("EVENT", &calib_evt, "slope/F:intercept/F:chi2/F:strip/I:layer/I:cham/I");
    
    for (int dduiter=0;dduiter<Nddu;dduiter++){
      for(int chamberiter=0; chamberiter<NChambers; chamberiter++){
        for (int cham=0;cham<NChambers;cham++){
	  if (cham !=chamberiter) continue;
	  
	  //get chamber ID from DB mapping        
	  int new_crateID = crateID[cham];
	  int new_dmbID   = dmbID[cham];
	  std::cout<<" Crate: "<<new_crateID<<" and DMB:  "<<new_dmbID<<std::endl;
	  map->crate_chamber(new_crateID,new_dmbID,&chamber_id,&chamber_num,&sector);
	  std::cout<<"Data is for chamber:: "<< chamber_id<<" in sector:  "<<sector<<std::endl;
	  
	  for (int layeriter=0; layeriter<LAYERS; layeriter++){
	    for (int stripiter=0; stripiter<STRIPS; stripiter++){
	      
	      for (int j=0; j<LAYERS; j++){//layer
		if (j != layeriter) continue;
		
		int layer_id=chamber_num+j+1;
		if(sector==-100)continue;
		cn->obj[layer_id].resize(size[cham]);
		for (int k=0; k<size[cham]; k++){//strip
		  if (k != stripiter) continue;
		  float sumOfX    = 0.0;
		  float sumOfY    = 0.0;
		  float sumOfXY   = 0.0;
		  float sumx2     = 0.0;
		  float gainSlope = 0.0;
		  float gainIntercept = 0.0;
		  float chi2      = 0.0;
		  
		  float charge[NUMBERPLOTTED]={22.4, 44.8, 67.2, 89.6, 112, 134.4, 156.8, 179.2, 201.6, 224.0};
		  
		  for(int ii=0; ii<NUMBERPLOTTED; ii++){//numbers    
		    sumOfX  += charge[ii];
		    sumOfY  += maxmodten[ii][cham][j][k];
		    sumOfXY += (charge[ii]*maxmodten[ii][cham][j][k]);
		    sumx2   += (charge[ii]*charge[ii]);
		  }
		 
		  //Fit parameters for straight line
		  gainSlope     = ((NUMBERPLOTTED*sumOfXY) - (sumOfX * sumOfY))/((NUMBERPLOTTED*sumx2) - (sumOfX*sumOfX));//k
		  gainIntercept = ((sumOfY*sumx2)-(sumOfX*sumOfXY))/((NUMBERPLOTTED*sumx2)-(sumOfX*sumOfX));//m
		  
		  for(int ii=0; ii<NUMBERPLOTTED; ii++){
		    chi2  += (maxmodten[ii][cham][j][k]-(gainIntercept+(gainSlope*charge[ii])))*(maxmodten[ii][cham][j][k]-(gainIntercept+(gainSlope*charge[ii])))/(NUMBERPLOTTED*NUMBERPLOTTED);
		  }
		  
		  std::cout <<"Chamber: "<<cham<<" Layer:   "<<j<<" Strip:   "<<k<<"  Slope:    "<<gainSlope <<"    Intercept:    "<<gainIntercept <<"        chi2 "<<chi2<<std::endl;
		  
		  calib_evt.slope     = gainSlope;
		  calib_evt.intercept = gainIntercept;
		  calib_evt.chi2      = chi2;
		  calib_evt.strip     = k;
		  calib_evt.layer     = j;
		  calib_evt.cham      = cham;
		  
		  calibtree.Fill();
		  
		  cn->obj[layer_id][k].resize(3);
		  cn->obj[layer_id][k][0] = gainSlope;
		  cn->obj[layer_id][k][1] = gainIntercept;
		  cn->obj[layer_id][k][2] = chi2;
		}//k loop
	      }//j loop
	    }//stripiter loop
	  }//layiter loop
	}//cham 
      }//chamberiter
    }//dduiter
    
    //send data to DB
    dbon->cdbon_last_run("gains",&run);
    std::cout<<"Last gains run "<<run<<" for run file "<<myname<<" saved "<<myTime<<std::endl;
    if(debug) dbon->cdbon_write(cn,"gains",run+1,myTime);
    calibfile.Write();
    calibfile.Close();
  }
  
 private:
  std::vector<int> newadc; 
  std::string chamber_id;
  int eventNumber,evt,chamber_num,sector,i_chamber,i_layer,reportedChambers;
  int fff,ret_code,length,strip,misMatch,NChambers,Nddu,run;
  time_t rawtime;
  int dmbID[CHAMBERS],crateID[CHAMBERS],size[CHAMBERS]; 
  float gainSlope,gainIntercept;
  float adcMax[DDU][CHAMBERS][LAYERS][STRIPS];
  float adcMean_max[DDU][CHAMBERS][LAYERS][STRIPS];
  float maxmodten[NUMMODTEN][CHAMBERS][LAYERS][STRIPS];
  float newGain[480];
  float newIntercept[480];
  float newChi2[480];
  int lines;
  std::ifstream filein;
  string PSet,name;
  bool debug;
};

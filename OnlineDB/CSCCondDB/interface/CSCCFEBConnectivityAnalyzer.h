/** 
 * Analyzer for calculating CFEB ADC counts for connectivity.
 * author S.Durkin, O.Boeriu 14/07/06 
 * runs over multiple DDUs
 * takes variable size chambers & layers 
 * produces histograms & ntuple 
 */

#include <iostream>
#include <time.h>
#include <sys/stat.h>	
#include <unistd.h>
#include <fstream>

#include "OnlineDB/CSCCondDB/interface/cscmap.h"
#include "OnlineDB/CSCCondDB/interface/CSCOnlineDB.h"
#include "CondFormats/CSCObjects/interface/CSCobject.h"
#include "TFile.h"
#include "TTree.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TCanvas.h"

class TCalibCFEBConnectEvt {
  public:
  Int_t strip;
  Int_t layer;
  Int_t cham;
  Int_t ddu;
  Float_t adcMax;
  Float_t adcMin;
  Float_t diff;
};

class CSCCFEBConnectivityAnalyzer : public edm::EDAnalyzer {
 public:
  explicit CSCCFEBConnectivityAnalyzer(edm::ParameterSet const& conf);
  virtual void analyze(edm::Event const& e, edm::EventSetup const& iSetup);
   
#define CHAMBERS_con 18
#define LAYERS_con 6
#define STRIPS_con 80
#define TIMEBINS_con 8
#define DDU_con 2
#define TOTALSTRIPS_con 480
#define TOTALEVENTS_con 320

  ~CSCCFEBConnectivityAnalyzer(){

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
    
    //get name of run file from .cfg and name root output after that
    string::size_type runNameStart = name.find("\"",0);
    string::size_type runNameEnd   = name.find("bin",0);
    string::size_type rootStart    = name.find("Crosstalk",0);
    int nameSize = runNameEnd+2-runNameStart;
    int myRootSize = rootStart-runNameStart+8;
    std::string myname= name.substr(runNameStart+1,nameSize);
    std::string myRootName= name.substr(runNameStart+1,myRootSize);
    std::string myRootEnd = "_conn.root";
    std::string runFile= myRootName;
    std::string myRootFileName = runFile+myRootEnd;
    const char *myNewName=myRootFileName.c_str();

    struct tm* clock;			    
    struct stat attrib;			    
    stat(myname.c_str(), &attrib);          
    clock = localtime(&(attrib.st_mtime));  
    std::string myTime=asctime(clock);

    //DB object and map
    //CSCobject *cn = new CSCobject();
    cscmap *map = new cscmap();
    //condbon *dbon = new condbon();


    //root ntuple
    TCalibCFEBConnectEvt calib_evt;
    TFile calibfile(myNewName, "RECREATE");
    TTree calibtree("Calibration","Connectivity");
    calibtree.Branch("EVENT", &calib_evt, "strip/I:layer/I:cham/I:ddu/I:adcMax/F:adcMin/F:diff/F");
   
    for (int iii=0; iii<Nddu; iii++){
      for (int i=0; i<NChambers; i++){
	
	//get chamber ID from DB mapping        
	int new_crateID = crateID[i];
	int new_dmbID   = dmbID[i];
	std::cout<<" Crate: "<<new_crateID<<" and DMB:  "<<new_dmbID<<std::endl;
	map->crate_chamber(new_crateID,new_dmbID,&chamber_id,&chamber_num,&sector);
	std::cout<<"Data is for chamber:: "<< chamber_id<<" in sector:  "<<sector<<std::endl;
	
	for (int j=0; j<LAYERS_con; j++){
	  for (int k=0; k<size[i]; k++){
	   
	    my_diff = adcMax[iii][i][j][k]-adcMin[iii][i][j][k];
	    std::cout<<"Chamber "<<i<<" Layer "<<j<<" Strip "<<k<<" diff "<<my_diff<<std::endl;

	    calib_evt.strip=k;
	    calib_evt.layer=j;
	    calib_evt.cham=i;
	    calib_evt.ddu=iii;
	    calib_evt.adcMin = adcMin[iii][i][j][k];
	    calib_evt.adcMax = adcMax[iii][i][j][k];
	    calib_evt.diff=my_diff;

	    calibtree.Fill();
	  }
	}
      }
    }
    calibfile.Write();    
    calibfile.Close();  
  }

 private:
  int eventNumber,evt,strip,misMatch,fff,ret_code,length,Nddu,myevt;
  int chamber,layer,reportedChambers,chamber_num,sector,record,NChambers ;
  int dmbID[CHAMBERS_con],crateID[CHAMBERS_con],size[CHAMBERS_con];
  float adcMin[DDU_con][CHAMBERS_con][LAYERS_con][STRIPS_con];
  float adcMax[DDU_con][CHAMBERS_con][LAYERS_con][STRIPS_con];
  float diff[DDU_con][CHAMBERS_con][LAYERS_con][STRIPS_con];
  std::vector<int> adc;
  std::string chamber_id;
  int lines;
  float my_diff;
  std::ifstream filein;
  string PSet,name;
  bool debug;
  int flag;

 //root ntuple
  TCalibCFEBConnectEvt calib_evt;
  TBranch *calibevt;
  TTree *calibtree;
  TFile *calibfile;
};



/** 
 * Analyzer for reading bin by bin ADC information
 * author S.Durkin, O.Boeriu 20/03/06 
 *   
 */

#include <iostream>
#include <time.h>
#include <sys/stat.h>	
#include <unistd.h>
#include <fstream>

#include "OnlineDB/CSCCondDB/interface/condbon.h"
#include "OnlineDB/CSCCondDB/interface/cscmap.h"
#include "OnlineDB/CSCCondDB/interface/AutoCorrMat.h"
#include "TFile.h"
#include "TTree.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TDirectory.h"
#include "TCanvas.h"

class TCalibNoiseMatrixEvt {
  public:
  Float_t elem[12];
  Int_t strip;
  Int_t layer;
  Int_t cham;
};

class CSCNoiseMatrixAnalyzer : public edm::EDAnalyzer {

 public:
  explicit CSCNoiseMatrixAnalyzer(edm::ParameterSet const& conf);
  virtual void analyze(edm::Event const& e, edm::EventSetup const& iSetup);
  
#define CHAMBERS_ma 468
#define LAYERS_ma 6
#define STRIPS_ma 80
#define DDU_ma 36

  ~CSCNoiseMatrixAnalyzer(){
    //get time of Run file for DB transfer
    filein.open("../test/CSCmatrix.cfg");
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
    
    //root ntuple
    TCalibNoiseMatrixEvt calib_evt;
    TFile calibfile("calibmatrix.root", "RECREATE");
    TTree calibtree("Calibration","NoiseMatrix");
    calibtree.Branch("EVENT", &calib_evt, "elem[12]/F:strip/I:layer/I:cham/I");
   
   //for (int myDDU; myDDU<Nddu; myDDU++){
     for (int i=0; i<NChambers; i++){
     
       //get chamber ID from DB mapping        
       int new_crateID = crateID[i];
       int new_dmbID   = dmbID[i];
       std::cout<<" Crate: "<<new_crateID<<" and DMB:  "<<new_dmbID<<std::endl;
       map->crate_chamber(new_crateID,new_dmbID,&chamber_id,&chamber_num,&sector);
       std::cout<<"Data is for chamber:: "<< chamber_id<<" in sector:  "<<sector<<std::endl;
       
       for (int j=0; j<LAYERS_ma; j++){
	 int layer_id=chamber_num+j+1;
	 if(sector==-100)continue;
	 cn->obj[layer_id].resize(size[i]);
	 
	 for (int k=0; k<size[i]; k++){
	   for (int max=0; max<12;max++){
	     tmp=cam[i].autocorrmat(j,k);
	     calib_evt.elem[0] = tmp[0];
	     calib_evt.elem[1] = tmp[1];
	     calib_evt.elem[2] = tmp[2];
	     calib_evt.elem[3] = tmp[3];
	     calib_evt.elem[4] = tmp[4];
	     calib_evt.elem[5] = tmp[5];
	     calib_evt.elem[6] = tmp[6];
	     calib_evt.elem[7] = tmp[7];
	     calib_evt.elem[8] = tmp[8];
	     calib_evt.elem[9] = tmp[9];
	     calib_evt.elem[10] = tmp[10];
	     calib_evt.elem[11] = tmp[11];
	     calib_evt.strip   = k;
	     calib_evt.layer   = j;
	     calib_evt.cham    = i;
	     
	     calibtree.Fill();
	     
	     std::cout<<"Chamber "<<i<<" Layer "<<j<<" strip "<<k<<" Matrix elements "<<tmp[max]<<std::endl;
	     if(tmp[max]>100) std::cout<<"Matrix elements out of range!"<<std::endl;
	     cn->obj[layer_id][k].resize(12);

	     cn->obj[layer_id][k][0] = tmp[0];
	     cn->obj[layer_id][k][1] = tmp[1];
	     cn->obj[layer_id][k][2] = tmp[3];
	     cn->obj[layer_id][k][3] = tmp[2];
	     cn->obj[layer_id][k][4] = tmp[4];
	     cn->obj[layer_id][k][5] = tmp[6];
	     cn->obj[layer_id][k][6] = tmp[5];
	     cn->obj[layer_id][k][7] = tmp[7];
	     cn->obj[layer_id][k][8] = tmp[9];
	     cn->obj[layer_id][k][9] = tmp[8];
	     cn->obj[layer_id][k][10] = tmp[10];
	     cn->obj[layer_id][k][11] = tmp[11];
	     
	   }
	 }
       }
     }
     //}//myDDU
     
     //send data to DB
     dbon->cdbon_last_run("noisematrix",&run);
     std::cout<<"run "<<run<<" for run file "<<myname<<" saved "<<myTime<<std::endl;
     if(debug) dbon->cdbon_write(cn,"noisematrix",run+1,myTime);

     calibfile.Write();
     calibfile.Close();
  }

 private:
 // variables persistent across events should be declared here.
 std::vector<int> adc;
 std::string chamber_id;
 int eventNumber,evt,strip,misMatch,NChambers,Nddu;
 int i_chamber,i_layer,reportedChambers,fff,ret_code,length,chamber_num,sector,run;
 int dmbID[CHAMBERS_ma],crateID[CHAMBERS_ma],size[CHAMBERS_ma];
 int lines;
 std::ifstream filein;
 string PSet,name;
 bool debug;
 float *tmp, corrmat[12];
 float newMatrix1[480];
 float newMatrix2[480];
 float newMatrix3[480];
 float newMatrix4[480];
 float newMatrix5[480];
 float newMatrix6[480];
 float newMatrix7[480];
 float newMatrix8[480];
 float newMatrix9[480];
 float newMatrix10[480];
 float newMatrix11[480];
 float newMatrix12[480];

 Chamber_AutoCorrMat cam[CHAMBERS_ma];
};

/** 
 * Analyzer for reading bin by bin ADC information
 * author S.Durkin, O.Boeriu 20/03/06 
 *   
 */

#include <iostream>
#include "CalibMuon/CSCCalibration/interface/condbc.h"
#include "CalibMuon/CSCCalibration/interface/cscmap.h"
#include "/afs/cern.ch/cms/external/lcg/external/root/5.08.00/slc3_ia32_gcc323/root/include/TFile.h"
#include "/afs/cern.ch/cms/external/lcg/external/root/5.08.00/slc3_ia32_gcc323/root/include/TTree.h"
#include "/afs/cern.ch/cms/external/lcg/external/root/5.08.00/slc3_ia32_gcc323/root/include/TH1F.h"
#include "/afs/cern.ch/cms/external/lcg/external/root/5.08.00/slc3_ia32_gcc323/root/include/TH2F.h"
#include "/afs/cern.ch/cms/external/lcg/external/root/5.08.00/slc3_ia32_gcc323/root/include/TDirectory.h"
#include "/afs/cern.ch/cms/external/lcg/external/root/5.08.00/slc3_ia32_gcc323/root/include/TCanvas.h"

class TCalibEvt {
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
  
#define CHAMBERS 9
#define LAYERS 6
#define STRIPS 80

  ~CSCNoiseMatrixAnalyzer(){
   
   condbc *cdb = new condbc();
   cscmap *map = new cscmap();
   
   //root ntuple
   TCalibEvt calib_evt;
   TFile calibfile("ntuples/calibmatrix.root", "RECREATE");
   TTree calibtree("Calibration","NoiseMatrix");
   calibtree.Branch("EVENT", &calib_evt, "elem[12]/F:strip/I:layer/I:cham/I");

   std::string test1="CSC_slice";
   std::string test2="elem33";
   std::string test3="elem34";
   std::string test4="elem44";
   std::string test5="elem35";
   std::string test6="elem45";
   std::string test7="elem55";
   std::string test8="elem46";
   std::string test9="elem56";
   std::string test10="elem66";
   std::string test11="elem57";
   std::string test12="elem67";
   std::string test13="elem77";
   std::string answer;
   
   //for (int myDDU; myDDU<Nddu; myDDU++){
     for (int i=0; i<NChambers; i++){
       for (int j=0; j<LAYERS; j++){
	 for (int k=0; k<size[i]; k++){
	   for (int max=0; max<12;max++){
	     fff = (j*size[i])+k;
	     tmp=cam[i].autocorrmat(j,k);
	     newMatrix1[fff] =tmp[0];
	     newMatrix2[fff] =tmp[1];
	     newMatrix3[fff] =tmp[2];
	     newMatrix4[fff] =tmp[3];
	     newMatrix5[fff] =tmp[4];
	     newMatrix6[fff] =tmp[5];
	     newMatrix7[fff] =tmp[6];
	     newMatrix8[fff] =tmp[7];
	     newMatrix9[fff] =tmp[8];
	     newMatrix10[fff]=tmp[9];
	     newMatrix11[fff]=tmp[10];
	     newMatrix12[fff]=tmp[11];
	     
	     calib_evt.elem[0] = newMatrix1[fff];
	     calib_evt.elem[1] = newMatrix2[fff];
	     calib_evt.elem[2] = newMatrix3[fff];
	     calib_evt.elem[3] = newMatrix4[fff];
	     calib_evt.elem[4] = newMatrix5[fff];
	     calib_evt.elem[5] = newMatrix6[fff];
	     calib_evt.elem[6] = newMatrix7[fff];
	     calib_evt.elem[7] = newMatrix8[fff];
	     calib_evt.elem[8] = newMatrix9[fff];
	     calib_evt.elem[9] = newMatrix10[fff];
	     calib_evt.elem[10] = newMatrix11[fff];
	     calib_evt.elem[11] = newMatrix12[fff];
	     calib_evt.strip   = k;
	     calib_evt.layer   = j;
	     calib_evt.cham    = i;
	     
	     calibtree.Fill();
	     
	     std::cout<<"Chamber "<<i<<" Layer "<<j<<" strip "<<k<<" Matrix elements "<<tmp[max]<<std::endl;
	     if(tmp[max]>100) std::cout<<"Matrix elements out of range!"<<std::endl;
	   }
	 }
       }
     
       //get chamber ID from Igor's mapping
       
       int new_crateID = crateID[i];
       int new_dmbID   = dmbID[i];
       std::cout<<"Here is crate: "<<new_crateID<<" and DMB:  "<<new_dmbID<<std::endl;
       map->crate_chamber(new_crateID,new_dmbID,&chamber_id,&chamber_num,&sector);
       
       std::cout<<" Above data is for chamber: "<< chamber_id<<" from sector "<<sector<<std::endl;
       
       std::cout<<" DO you want to send constants to DB? "<<std::endl;
       std::cout<<" Please answer y or n for EACH chamber present! "<<std::endl;
       
       std::cin>>answer;
       if(answer=="y"){
	 //SEND CONSTANTS TO DB
	 if(eventNumber != 1000){std::cout<<"Number of events not as expected! "<<eventNumber<<std::endl; continue;}
	 
	 cdb->cdb_write(test1,chamber_id,chamber_num,test2, size[i]*6, newMatrix1, 6, &ret_code);
	 cdb->cdb_write(test1,chamber_id,chamber_num,test3, size[i]*6, newMatrix2, 6, &ret_code);
	 cdb->cdb_write(test1,chamber_id,chamber_num,test4, size[i]*6, newMatrix3, 6, &ret_code);
	 cdb->cdb_write(test1,chamber_id,chamber_num,test5, size[i]*6, newMatrix4, 6, &ret_code);
	 cdb->cdb_write(test1,chamber_id,chamber_num,test6, size[i]*6, newMatrix5, 6, &ret_code);
	 cdb->cdb_write(test1,chamber_id,chamber_num,test7, size[i]*6, newMatrix6, 6, &ret_code);
	 cdb->cdb_write(test1,chamber_id,chamber_num,test8, size[i]*6, newMatrix7, 6, &ret_code);
	 cdb->cdb_write(test1,chamber_id,chamber_num,test9, size[i]*6, newMatrix8, 6, &ret_code);
	 cdb->cdb_write(test1,chamber_id,chamber_num,test10,size[i]*6, newMatrix9, 6, &ret_code);
	 cdb->cdb_write(test1,chamber_id,chamber_num,test11,size[i]*6, newMatrix10,6, &ret_code);
	 cdb->cdb_write(test1,chamber_id,chamber_num,test12,size[i]*6, newMatrix11,6, &ret_code);
	 cdb->cdb_write(test1,chamber_id,chamber_num,test13,size[i]*6, newMatrix12,6, &ret_code);
       }else{
	 std::cout<<" NO data was sent!!! "<<std::endl;
       }
     }
     //}
   calibfile.Write();
   calibfile.Close();
  }

 private:
 // variables persistent across events should be declared here.
 std::vector<int> adc; 
 std::string chamber_id;
 int eventNumber,evt,strip,misMatch,NChambers,Nddu;
 int i_chamber,i_layer,reportedChambers,fff,ret_code,length,chamber_num,sector;
 int dmbID[CHAMBERS],crateID[CHAMBERS],size[CHAMBERS];
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
 
 Chamber_AutoCorrMat cam[CHAMBERS];
};

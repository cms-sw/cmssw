/** 
 * Analyzer for reading bin by bin ADC information
 * author S.Durkin, O.Boeriu 20/03/06 
 *   
 */

#include <iostream>
#include "CalibMuon/CSCCalibration/interface/condbc.h"
#include "CalibMuon/CSCCalibration/interface/cscmap.h"

class CSCNoiseMatrixAnalyzer : public edm::EDAnalyzer {
 public:
  explicit CSCNoiseMatrixAnalyzer(edm::ParameterSet const& conf);
  virtual void analyze(edm::Event const& e, edm::EventSetup const& iSetup);
  
#define CHAMBERS 5
#define LAYERS 6
#define STRIPS 80

  ~CSCNoiseMatrixAnalyzer(){
   
   condbc *cdb = new condbc();
   cscmap *map = new cscmap();

   std::string test1="CSC_slice";
   std::string test2="elem33";
   std::string test3="elem34";
   std::string test4="elem35";
   std::string test5="elem44";
   std::string test6="elem45";
   std::string test7="elem46";
   std::string test8="elem55";
   std::string test9="elem56";
   std::string test10="elem57";
   std::string test11="elem66";
   std::string test12="elem67";
   std::string test13="elem77";
   std::string answer;
   
   for (int i=0; i<CHAMBERS; i++){
     for (int j=0; j<LAYERS; j++){
       for (int k=0; k<STRIPS; k++){
	 for (int max=0; max<12;max++){
	   fff = (j*80)+k;
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
	   
	   std::cout<<"Chamber "<<i<<" Layer "<<j<<" strip "<<k<<" Matrix elements "<<tmp[max]<<std::endl;
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
       
       cdb->cdb_write(test1,chamber_id,chamber_num,test2, 480, newMatrix1, 2, &ret_code);
       cdb->cdb_write(test1,chamber_id,chamber_num,test3, 480, newMatrix2, 2, &ret_code);
       cdb->cdb_write(test1,chamber_id,chamber_num,test4, 480, newMatrix3, 2, &ret_code);
       cdb->cdb_write(test1,chamber_id,chamber_num,test5, 480, newMatrix4, 2, &ret_code);
       cdb->cdb_write(test1,chamber_id,chamber_num,test6, 480, newMatrix5, 2, &ret_code);
       cdb->cdb_write(test1,chamber_id,chamber_num,test7, 480, newMatrix6, 2, &ret_code);
       cdb->cdb_write(test1,chamber_id,chamber_num,test8, 480, newMatrix7, 2, &ret_code);
       cdb->cdb_write(test1,chamber_id,chamber_num,test9, 480, newMatrix8, 2, &ret_code);
       cdb->cdb_write(test1,chamber_id,chamber_num,test10,480, newMatrix9, 2, &ret_code);
       cdb->cdb_write(test1,chamber_id,chamber_num,test11,480, newMatrix10,2, &ret_code);
       cdb->cdb_write(test1,chamber_id,chamber_num,test12,480, newMatrix11,2, &ret_code);
       cdb->cdb_write(test1,chamber_id,chamber_num,test13,480, newMatrix12,2, &ret_code);
     }else{
       std::cout<<" NO data was sent!!! "<<std::endl;
     }
   }

   
 }
 
 private:
 // variables persistent across events should be declared here.
 std::vector<int> adc; 
 std::string chamber_id;
 int eventNumber,evt,strip,misMatch;
 int i_chamber,i_layer,reportedChambers,fff,ret_code,length,chamber_num,sector;
 int dmbID[CHAMBERS],crateID[CHAMBERS];
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
 
 Chamber_AutoCorrMat cam[5];
};

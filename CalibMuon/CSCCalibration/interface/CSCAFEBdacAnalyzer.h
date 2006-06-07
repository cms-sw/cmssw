/** 
 * Analyzer for reading AFEB thresholds
 * author O.Boeriu 9/05/06 
 *   
 */

#include <iostream>
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CalibMuon/CSCCalibration/interface/condbc.h"
#include "CalibMuon/CSCCalibration/interface/cscmap.h"

class CSCAFEBdacAnalyzer : public edm::EDAnalyzer {
 public:
  explicit CSCAFEBdacAnalyzer(edm::ParameterSet const& conf);
  virtual void analyze(edm::Event const& e, edm::EventSetup const& iSetup);
  
#define CHAMBERS 9
#define LAYERS 6
#define WIRES 64
#define TOTALWIRES 384
#define DDU 2

  ~CSCAFEBdacAnalyzer(){
 
    //create array (480 entries) for database transfer
    condbc *cdb = new condbc();
    cscmap *map = new cscmap();

    for(int myDDU=0;myDDU<Nddu;myDDU++){

      for(int myChamber=0; myChamber<NChambers; myChamber++){

	std::string test1="CSC_slice";
	std::string test2="afeb_thresholds";
	//std::string test3="ped_rms";
	//std::string test4="peak_spread";
	//std::string test5="pulse_shape";
	std::string answer;

	for (int ii=0;ii<Nddu;ii++){
	  if (myDDU !=ii) continue;
	  for (int i=0; i<NChambers; i++){
	    if (myChamber !=i) continue;
	    
	    for (int j=0; j<LAYERS; j++){
	      for (int k=0; k<WIRES; k++){
		fff = (j*80)+k;

	      }
	    }
	  }
	}//DDU
	
	
	//get chamber ID from Igor's mapping
	
	int new_crateID = crateID[myChamber];
	int new_dmbID   = dmbID[myChamber];
	std::cout<<" Crate: "<<new_crateID<<" and DMB:  "<<new_dmbID<<std::endl;
	map->crate_chamber(new_crateID,new_dmbID,&chamber_id,&chamber_num,&sector);
	std::cout<<" Above data is for chamber: "<< chamber_id<<" from sector "<<sector<<std::endl;
	
	std::cout<<" DO you want to send constants to DB? "<<std::endl;
	std::cout<<" Please answer y or n for EACH chamber present! "<<std::endl;
	
	std::cin>>answer;
	if(answer=="y"){
	  //SEND CONSTANTS TO DB
	  //cdb->cdb_write(test1,chamber_id,chamber_num,test2,480, newPed,    6, &ret_code);
	  //cdb->cdb_write(test1,chamber_id,chamber_num,test3,480, newRMS,    6, &ret_code);
	  //cdb->cdb_write(test1,chamber_id,chamber_num,test4,480, newPeakRMS,6, &ret_code);
	  //cdb->cdb_write(test1,chamber_id,chamber_num,test5,480, newSumFive,6, &ret_code);
	  
	  std::cout<<" Your results were sent to DB !!! "<<std::endl;
	}else{
	  std::cout<<" NO data was sent!!! "<<std::endl;
	}
      }
      
    }

  }

 private:
  // variables persistent across events should be declared here.
  //

  int eventNumber,evt,misMatch,fff,ret_code,NChambers,Nddu,event;
  int wireGroup,wireTBin,length,i_chamber,i_layer,reportedChambers,chamber_num,sector; 
  int dmbID[CHAMBERS],crateID[CHAMBERS];
  std::string chamber_id;
};

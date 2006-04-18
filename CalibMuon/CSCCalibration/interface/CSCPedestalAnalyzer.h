/** 
 * Analyzer for reading pedestal information
 * author O.Boeriu 18/03/06 
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

#include </afs/cern.ch/cms/external/lcg/external/root/5.08.00/slc3_ia32_gcc323/root/include/TFile.h>
#include "/afs/cern.ch/cms/external/lcg/external/root/5.08.00/slc3_ia32_gcc323/root/include/TTree.h"
#include "/afs/cern.ch/cms/external/lcg/external/root/5.08.00/slc3_ia32_gcc323/root/include/TH1F.h"
#include "/afs/cern.ch/cms/external/lcg/external/root/5.08.00/slc3_ia32_gcc323/root/include/TH2F.h"
#include "/afs/cern.ch/cms/external/lcg/external/root/5.08.00/slc3_ia32_gcc323/root/include/TDirectory.h"

class TCalibEvt {
  public:
  Float_t pedMean;
  Float_t pedRMS;
  Int_t strip;
  Int_t layer;
  Int_t cham;
};

class CSCPedestalAnalyzer : public edm::EDAnalyzer {
 public:
  explicit CSCPedestalAnalyzer(edm::ParameterSet const& conf);
  virtual void analyze(edm::Event const& e, edm::EventSetup const& iSetup);
  
#define CHAMBERS 5
#define LAYERS 6
#define STRIPS 80

  int evt;
  float ped;
  std::vector<int> newadc;
  std::vector<int> adc;
  std::string chamber_id;
  float pedMean,time,max,max1;
  int pedSum, strip, misMatch;
  int dmbID[CHAMBERS],crateID[CHAMBERS],chamber_num,sector;
  int i_chamber,i_layer,reportedChambers ;
  int fff,ret_code;
  float aPeak,sumFive;
  int length;
  float meanPedestal ,meanPeak,meanPeakSquare, meanPedestalSquare,theRMS,thePedestal,theRSquare ;
  float thePeak,thePeakRMS,theSumFive;

  //definition of arrays
  float arrayOfPed[CHAMBERS][LAYERS][STRIPS];
  float arrayOfPedSquare[CHAMBERS][LAYERS][STRIPS];
  float arrayPed[CHAMBERS][LAYERS][STRIPS];
  float arrayPeak[CHAMBERS][LAYERS][STRIPS];
  float arrayOfPeak[CHAMBERS][LAYERS][STRIPS];
  float arrayOfPeakSquare[CHAMBERS][LAYERS][STRIPS];
  float arraySumFive[CHAMBERS][LAYERS][STRIPS];
  
  float newPed[480];
  float newRMS[480];
  float newPeakRMS[480];
  float newPeak[480];
  float newSumFive[480];

  ~CSCPedestalAnalyzer(){
 
    //create array (480 entries) for database transfer
    condbc *cdb = new condbc();
    cscmap *map = new cscmap();
      
    //root ntuple
    TCalibEvt calib_evt;
    TFile calibfile("calibpedestal.root", "RECREATE");
    TDirectory * dir1 = calibfile.mkdir("histo");
    calibfile.cd("histo");
    TH2F * ped11 = new TH2F("Ped11 vs strip","pedestal in 1st layer/chamber",80,0.,80.,80,500.,700.);
    TH2F * ped12 = new TH2F("Ped12 vs strip","pedestal in 2nd layer 1st chamber",80,0.,80.,80,500.,700.);
    TH2F * ped13 = new TH2F("Ped13 vs strip","pedestal in 3rd layer/chamber",80,0.,80.,80,500.,700.);
    TH2F * ped14 = new TH2F("Ped14 vs strip","pedestal in 4th layer 1st chamber",80,0.,80.,80,500.,700.);
    TH2F * ped15 = new TH2F("Ped15 vs strip","pedestal in 5th layer/chamber",80,0.,80.,80,500.,750.);
    TH2F * ped16 = new TH2F("Ped16 vs strip","pedestal in 6th layer 1st chamber",80,0.,80.,80,500.,700.);

    calibfile.cd();
    TTree calibtree("Calibration","Pedestal");
    calibtree.Branch("EVENT", &calib_evt, "pedMean/F:pedRMS/F:strip/I:layer/I:cham/I");

    for(int myChamber=0; myChamber<CHAMBERS; myChamber++){
      meanPedestal = 0.0,meanPeak=0.0,meanPeakSquare=0.;
      meanPedestalSquare = 0.;
      theRMS = 0.;
      thePedestal =0.;
      theRSquare = 0.;
      thePeak =0.0,thePeakRMS=0.0;
      theSumFive=0.0;
      
      std::string test1="CSC_slice";
      std::string test2="pedestal";
      std::string test3="ped_rms";
      std::string test4="peak_spread";
      std::string test5="pulse_shape";
      std::string answer;
      
      for (int i=0; i<CHAMBERS; i++){
	if (myChamber !=i) continue;
	
	for (int j=0; j<LAYERS; j++){
	  for (int k=0; k<STRIPS; k++){
	    fff = (j*80)+k;
	    thePedestal  = arrayPed[i][j][k];
	    meanPedestal = arrayOfPed[i][j][k] / evt;
	    newPed[fff]  = meanPedestal;
	    meanPedestalSquare = arrayOfPedSquare[i][j][k] / evt;
	    theRMS       = sqrt(abs(meanPedestalSquare - meanPedestal*meanPedestal));
	    newRMS[fff]  = theRMS;
	    theRSquare   = (thePedestal-meanPedestal)*(thePedestal-meanPedestal)/(theRMS*theRMS*theRMS*theRMS);
	    
	    calib_evt.pedMean = meanPedestal;
	    calib_evt.pedRMS  = theRMS/1.;
	    calib_evt.strip   = k;
	    calib_evt.layer   = j;
	    calib_evt.cham    = i;

	    if(i==0 && j==0){
	      ped11->Fill(k,meanPedestal);
	    }
	    
	    if(i==0 && j==1){
	      ped12->Fill(k,meanPedestal);
	    }
	    if(i==0 && j==2){
	      ped13->Fill(k,meanPedestal);
	    }
	    
	    if(i==0 && j==3){
	      ped14->Fill(k,meanPedestal);
	    }
	    if(i==0 && j==4){
	      ped15->Fill(k,meanPedestal);
	    }
	    
	    if(i==0 && j==5){
	      ped16->Fill(k,meanPedestal);
	    }

	    thePeak = arrayPeak[i][j][k];
	    meanPeak = arrayOfPeak[i][j][k] / evt;
	    meanPeakSquare = arrayOfPeakSquare[i][j][k] / evt;
	    thePeakRMS = sqrt(abs(meanPeakSquare - meanPeak*meanPeak));
	    newPeakRMS[fff] = thePeakRMS;
	    newPeak[fff] = thePeak;
	    
	    theSumFive = arraySumFive[i][j][k];
	    newSumFive[fff]=theSumFive;

	    calibtree.Fill();	  	    
	    std::cout <<" chamber "<<i<<" layer "<<j<<" strip "<<fff<<"  ped "<<newPed[fff]<<" RMS "<<newRMS[fff]<<" peakADC "<<newPeak[fff]<<" Peak RMS "<<newPeakRMS[fff]<<" Sum_of_four/apeak "<<newSumFive[fff]<<std::endl;
	  }
	}
      }
 
     
      
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
	cdb->cdb_write(test1,chamber_id,chamber_num,test2,480, newPed, 2, &ret_code); 
	cdb->cdb_write(test1,chamber_id,chamber_num,test3,480, newRMS, 2, &ret_code);
	cdb->cdb_write(test1,chamber_id,chamber_num,test4,480, newPeakRMS,2, &ret_code);
	cdb->cdb_write(test1,chamber_id,chamber_num,test5,480, newSumFive,2, &ret_code);
	
	std::cout<<" Your results were sent to DB !!! "<<std::endl;
      }else{
	std::cout<<" NO data was sent!!! "<<std::endl;
      }
    }
    calibfile.Write();
    calibfile.Close();
  }
  
  
   private:
  // variables persistent across events should be declared here.
  //
  int eventNumber;
  
};

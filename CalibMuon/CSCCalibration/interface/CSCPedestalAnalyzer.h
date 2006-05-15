/** 
 * Analyzer for reading pedestal information
 * author O.Boeriu 27/04/06 
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
  
#define CHAMBERS 9
#define LAYERS 6
#define STRIPS 80
#define TOTALSTRIPS 480
#define DDU 2

  ~CSCPedestalAnalyzer(){
 
    //create array for database transfer
    condbc *cdb = new condbc();
    cscmap *map = new cscmap();
      
    //root ntuple
    TCalibEvt calib_evt;
    TFile calibfile("ntuples/calibpedestal.root", "RECREATE");
    TDirectory * dir1 = calibfile.mkdir("histos");
    calibfile.cd("histos");
    TCanvas * c1;
    c1 = new TCanvas("c1","PEDESTAL vs LAYER CHAMBER=0",200, 10, 1000, 800);
    c1->Divide(1,6);
   
    c1->cd(1);
    TH2F * ped11 = new TH2F("Ped11","ped layer=1 ",80,0.,80.,80,500.,800.);
    c1->cd(2);
    TH2F * ped12 = new TH2F("Ped12","ped layer=2 ",80,0.,80.,80,500.,800.);
    c1->cd(3);
    TH2F * ped13 = new TH2F("Ped13","ped layer=3 ",80,0.,80.,80,500.,800.);
    c1->cd(4);
    TH2F * ped14 = new TH2F("Ped14","ped layer=4 ",80,0.,80.,80,500.,800.);
    c1->cd(5);
    TH2F * ped15 = new TH2F("Ped15","ped layer=5 ",80,0.,80.,80,500.,800.);
    c1->cd(6);
    TH2F * ped16 = new TH2F("Ped16","ped layer=6 ",80,0.,80.,80,500.,800.);
    TH2F * ped21 = new TH2F("Ped21","ped layer=1 ",80,0.,80.,80,500.,800.);
    TH2F * ped22 = new TH2F("Ped22","ped layer=2 ",80,0.,80.,80,500.,800.);
    TH2F * ped23 = new TH2F("Ped23","ped layer=3 ",80,0.,80.,80,500.,800.);
    TH2F * ped24 = new TH2F("Ped24","ped layer=4 ",80,0.,80.,80,500.,800.);
    TH2F * ped25 = new TH2F("Ped25","ped layer=5 ",80,0.,80.,80,500.,800.);
    TH2F * ped26 = new TH2F("Ped26","ped layer=6 ",80,0.,80.,80,500.,800.);
    TH2F * ped31 = new TH2F("Ped31","ped layer=1 ",80,0.,80.,80,500.,800.);
    TH2F * ped32 = new TH2F("Ped32","ped layer=2 ",80,0.,80.,80,500.,800.);
    TH2F * ped33 = new TH2F("Ped33","ped layer=3 ",80,0.,80.,80,500.,800.);
    TH2F * ped34 = new TH2F("Ped34","ped layer=4 ",80,0.,80.,80,500.,800.);
    TH2F * ped35 = new TH2F("Ped35","ped layer=5 ",80,0.,80.,80,500.,800.);
    TH2F * ped36 = new TH2F("Ped36","ped layer=6 ",80,0.,80.,80,500.,800.);
    TH2F * ped41 = new TH2F("Ped41","ped layer=1 ",80,0.,80.,80,500.,800.);
    TH2F * ped42 = new TH2F("Ped42","ped layer=2 ",80,0.,80.,80,500.,800.);
    TH2F * ped43 = new TH2F("Ped43","ped layer=3 ",80,0.,80.,80,500.,800.);
    TH2F * ped44 = new TH2F("Ped44","ped layer=4 ",80,0.,80.,80,500.,800.);
    TH2F * ped45 = new TH2F("Ped45","ped layer=5 ",80,0.,80.,80,500.,800.);
    TH2F * ped46 = new TH2F("Ped46","ped layer=6 ",80,0.,80.,80,500.,800.);
    TH2F * ped51 = new TH2F("Ped51","ped layer=1 ",80,0.,80.,80,500.,800.);
    TH2F * ped52 = new TH2F("Ped52","ped layer=2 ",80,0.,80.,80,500.,800.);
    TH2F * ped53 = new TH2F("Ped53","ped layer=3 ",80,0.,80.,80,500.,800.);
    TH2F * ped54 = new TH2F("Ped54","ped layer=4 ",80,0.,80.,80,500.,800.);
    TH2F * ped55 = new TH2F("Ped55","ped layer=5 ",80,0.,80.,80,500.,800.);
    TH2F * ped56 = new TH2F("Ped56","ped layer=6 ",80,0.,80.,80,500.,800.);
    

    TH2F * RMS11 = new TH2F("RMS11","RMS in layer=1",80,0.,80.,80,0.,10.);
    TH2F * RMS12 = new TH2F("RMS12","RMS in layer=2",80,0.,80.,80,0.,10.);
    TH2F * RMS13 = new TH2F("RMS13","RMS in layer=3",80,0.,80.,80,0.,10.);
    TH2F * RMS14 = new TH2F("RMS14","RMS in layer=4",80,0.,80.,80,0.,10.);
    TH2F * RMS15 = new TH2F("RMS15","RMS in layer=5",80,0.,80.,80,0.,10.);
    TH2F * RMS16 = new TH2F("RMS16","RMS in layer=6",80,0.,80.,80,0.,10.);
    TH2F * RMS21 = new TH2F("RMS21","RMS in layer=1",80,0.,80.,80,0.,10.);
    TH2F * RMS22 = new TH2F("RMS22","RMS in layer=2",80,0.,80.,80,0.,10.);
    TH2F * RMS23 = new TH2F("RMS23","RMS in layer=3",80,0.,80.,80,0.,10.);
    TH2F * RMS24 = new TH2F("RMS24","RMS in layer=4",80,0.,80.,80,0.,10.);
    TH2F * RMS25 = new TH2F("RMS25","RMS in layer=5",80,0.,80.,80,0.,10.);
    TH2F * RMS26 = new TH2F("RMS26","RMS in layer=6",80,0.,80.,80,0.,10.);
    TH2F * RMS31 = new TH2F("RMS31","RMS in layer=1",80,0.,80.,80,0.,10.);
    TH2F * RMS32 = new TH2F("RMS32","RMS in layer=2",80,0.,80.,80,0.,10.);
    TH2F * RMS33 = new TH2F("RMS33","RMS in layer=3",80,0.,80.,80,0.,10.);
    TH2F * RMS34 = new TH2F("RMS34","RMS in layer=4",80,0.,80.,80,0.,10.);
    TH2F * RMS35 = new TH2F("RMS35","RMS in layer=5",80,0.,80.,80,0.,10.);
    TH2F * RMS36 = new TH2F("RMS36","RMS in layer=6",80,0.,80.,80,0.,10.);
    TH2F * RMS41 = new TH2F("RMS41","RMS in layer=1",80,0.,80.,80,0.,10.);
    TH2F * RMS42 = new TH2F("RMS42","RMS in layer=2",80,0.,80.,80,0.,10.);
    TH2F * RMS43 = new TH2F("RMS43","RMS in layer=3",80,0.,80.,80,0.,10.);
    TH2F * RMS44 = new TH2F("RMS44","RMS in layer=4",80,0.,80.,80,0.,10.);
    TH2F * RMS45 = new TH2F("RMS45","RMS in layer=5",80,0.,80.,80,0.,10.);
    TH2F * RMS46 = new TH2F("RMS46","RMS in layer=6",80,0.,80.,80,0.,10.);
    TH2F * RMS51 = new TH2F("RMS51","RMS in layer=1",80,0.,80.,80,0.,10.);
    TH2F * RMS52 = new TH2F("RMS52","RMS in layer=2",80,0.,80.,80,0.,10.);
    TH2F * RMS53 = new TH2F("RMS53","RMS in layer=3",80,0.,80.,80,0.,10.);
    TH2F * RMS54 = new TH2F("RMS54","RMS in layer=4",80,0.,80.,80,0.,10.);
    TH2F * RMS55 = new TH2F("RMS55","RMS in layer=5",80,0.,80.,80,0.,10.);
    TH2F * RMS56 = new TH2F("RMS56","RMS in layer=6",80,0.,80.,80,0.,10.);

    ped11->SetMarkerStyle(20);
    ped12->SetMarkerStyle(20);
    ped13->SetMarkerStyle(20);
    ped14->SetMarkerStyle(20);
    ped15->SetMarkerStyle(20);
    ped16->SetMarkerStyle(20);
    ped21->SetMarkerStyle(20);
    ped22->SetMarkerStyle(20);
    ped23->SetMarkerStyle(20);
    ped24->SetMarkerStyle(20);
    ped25->SetMarkerStyle(20);
    ped26->SetMarkerStyle(20);
    ped31->SetMarkerStyle(20);
    ped32->SetMarkerStyle(20);
    ped33->SetMarkerStyle(20);
    ped34->SetMarkerStyle(20);
    ped35->SetMarkerStyle(20);
    ped36->SetMarkerStyle(20);
    ped41->SetMarkerStyle(20);
    ped42->SetMarkerStyle(20);
    ped43->SetMarkerStyle(20);
    ped44->SetMarkerStyle(20);
    ped45->SetMarkerStyle(20);
    ped46->SetMarkerStyle(20);
    ped51->SetMarkerStyle(20);
    ped52->SetMarkerStyle(20);
    ped53->SetMarkerStyle(20);
    ped54->SetMarkerStyle(20);
    ped55->SetMarkerStyle(20);
    ped56->SetMarkerStyle(20);
    RMS11->SetMarkerStyle(20);
    RMS12->SetMarkerStyle(20);
    RMS13->SetMarkerStyle(20);
    RMS14->SetMarkerStyle(20);
    RMS15->SetMarkerStyle(20);
    RMS16->SetMarkerStyle(20);
    RMS21->SetMarkerStyle(20);
    RMS22->SetMarkerStyle(20);
    RMS23->SetMarkerStyle(20);
    RMS24->SetMarkerStyle(20);
    RMS25->SetMarkerStyle(20);
    RMS26->SetMarkerStyle(20);
    RMS31->SetMarkerStyle(20);
    RMS32->SetMarkerStyle(20);
    RMS33->SetMarkerStyle(20);
    RMS34->SetMarkerStyle(20);
    RMS35->SetMarkerStyle(20);
    RMS36->SetMarkerStyle(20);
    RMS41->SetMarkerStyle(20);
    RMS42->SetMarkerStyle(20);
    RMS43->SetMarkerStyle(20);
    RMS44->SetMarkerStyle(20);
    RMS45->SetMarkerStyle(20);
    RMS46->SetMarkerStyle(20);
    RMS51->SetMarkerStyle(20);
    RMS52->SetMarkerStyle(20);
    RMS53->SetMarkerStyle(20);
    RMS54->SetMarkerStyle(20);
    RMS55->SetMarkerStyle(20);
    RMS56->SetMarkerStyle(20);

    calibfile.cd();
    TTree calibtree("Calibration","Pedestal");
    calibtree.Branch("EVENT", &calib_evt, "pedMean/F:pedRMS/F:strip/I:layer/I:cham/I");

    for(int myDDU=0;myDDU<Nddu;myDDU++){

      for(int myChamber=0; myChamber<NChambers; myChamber++){
	meanPedestal = 0.0,meanPeak=0.0,meanPeakSquare=0.;
	meanPedestalSquare = 0.;
	theRMS      =0.0;
	thePedestal =0.0;
	theRSquare  =0.0;
	thePeak     =0.0;
	thePeakRMS  =0.0;
	theSumFive  =0.0;
	
	std::string test1="CSC_slice";
	std::string test2="pedestal";
	std::string test3="ped_rms";
	std::string test4="peak_spread";
	std::string test5="pulse_shape";
	std::string answer;
	
	for (int ii=0;ii<Nddu;ii++){
	  if (myDDU !=ii) continue;
	  for (int i=0; i<NChambers; i++){
	    if (myChamber !=i) continue;
	    
	    for (int j=0; j<LAYERS; j++){
	      for (int k=0; k<size[i]; k++){
		fff = (j*size[i])+k;
		thePedestal  = arrayPed[ii][i][j][k];
		meanPedestal = arrayOfPed[ii][i][j][k]/evt;
		newPed[fff]  = meanPedestal;
		meanPedestalSquare = arrayOfPedSquare[ii][i][j][k] / evt;
		theRMS       = sqrt(abs(meanPedestalSquare - meanPedestal*meanPedestal));
		newRMS[fff]  = theRMS;
		theRSquare   = (thePedestal-meanPedestal)*(thePedestal-meanPedestal)/(theRMS*theRMS*theRMS*theRMS);
		
		calib_evt.pedMean = newPed[fff];
		calib_evt.pedRMS  = newRMS[fff];
		calib_evt.strip   = k;
		calib_evt.layer   = j;
		calib_evt.cham    = i;
		
		c1->cd(1);
		if(i==0 && j==0){
		  ped11->Fill(k,meanPedestal);
		  RMS11->Fill(k,theRMS);
		}
		ped11->Draw();
		c1->cd(2);
		if(i==0 && j==1){
		  ped12->Fill(k,meanPedestal);
		  RMS12->Fill(k,theRMS);
		}
		ped12->Draw();
		
		if(i==0 && j==2){
		  ped13->Fill(k,meanPedestal);
		  RMS13->Fill(k,theRMS);
		}
		
		if(i==0 && j==3){
		  ped14->Fill(k,meanPedestal);
		  RMS14->Fill(k,theRMS);
		}
		if(i==0 && j==4){
		  ped15->Fill(k,meanPedestal);
		  RMS15->Fill(k,theRMS);
		}
		
		if(i==0 && j==5){
		  ped16->Fill(k,meanPedestal);
		  RMS16->Fill(k,theRMS);
		}
		
		if(i==1 && j==0){
		  ped21->Fill(k,meanPedestal);
		  RMS21->Fill(k,theRMS);
		}
		
		if(i==1 && j==1){
		  ped22->Fill(k,meanPedestal);
		  RMS22->Fill(k,theRMS);
		}
		if(i==1 && j==2){
		  ped23->Fill(k,meanPedestal);
		  RMS23->Fill(k,theRMS);
		}
		
		if(i==1 && j==3){
		  ped24->Fill(k,meanPedestal);
		  RMS24->Fill(k,theRMS);
		}
		if(i==1 && j==4){
		  ped25->Fill(k,meanPedestal);
		  RMS25->Fill(k,theRMS);
		}
		
		if(i==1 && j==5){
		  ped26->Fill(k,meanPedestal);
		  RMS26->Fill(k,theRMS);
		}
		
		if(i==2 && j==0){
		  ped31->Fill(k,meanPedestal);
		  RMS31->Fill(k,theRMS);
		}
		
		if(i==2 && j==1){
		  ped32->Fill(k,meanPedestal);
		  RMS32->Fill(k,theRMS);
		}
		if(i==2 && j==2){
		  ped33->Fill(k,meanPedestal);
		  RMS33->Fill(k,theRMS);
		}
		
		if(i==2 && j==3){
		  ped34->Fill(k,meanPedestal);
		  RMS34->Fill(k,theRMS);
		}
		if(i==2 && j==4){
		  ped35->Fill(k,meanPedestal);
		  RMS35->Fill(k,theRMS);
		}
		
		if(i==2 && j==5){
		  ped36->Fill(k,meanPedestal);
		  RMS36->Fill(k,theRMS);
		}
		
		if(i==3 && j==0){
		  ped41->Fill(k,meanPedestal);
		  RMS41->Fill(k,theRMS);
		}
		
		if(i==3 && j==1){
		  ped42->Fill(k,meanPedestal);
		  RMS42->Fill(k,theRMS);
		}
		if(i==3 && j==2){
		  ped43->Fill(k,meanPedestal);
		  RMS43->Fill(k,theRMS);
		}
		
		if(i==3 && j==3){
		  ped44->Fill(k,meanPedestal);
		  RMS44->Fill(k,theRMS);
		}
		if(i==3 && j==4){
		  ped45->Fill(k,meanPedestal);
		  RMS45->Fill(k,theRMS);
		}
	    
		if(i==3 && j==5){
		  ped46->Fill(k,meanPedestal);
		  RMS46->Fill(k,theRMS);
		}
		
		if(i==4 && j==0){
		  ped51->Fill(k,meanPedestal);
		  RMS51->Fill(k,theRMS);
		}
		
		if(i==4 && j==1){
		  ped52->Fill(k,meanPedestal);
		  RMS52->Fill(k,theRMS);
		}
		if(i==4 && j==2){
		  ped53->Fill(k,meanPedestal);
		  RMS53->Fill(k,theRMS);
		}
		
		if(i==4 && j==3){
		  ped54->Fill(k,meanPedestal);
		  RMS54->Fill(k,theRMS);
		}
		if(i==4 && j==4){
		  ped55->Fill(k,meanPedestal);
		  RMS55->Fill(k,theRMS);
		}
		
		if(i==4 && j==5){
		  ped56->Fill(k,meanPedestal);
		  RMS56->Fill(k,theRMS);
		}
		
		thePeak = arrayPeak[ii][i][j][k];
		meanPeak = arrayOfPeak[ii][i][j][k] / evt;
		meanPeakSquare = arrayOfPeakSquare[ii][i][j][k] / evt;
		thePeakRMS = sqrt(abs(meanPeakSquare - meanPeak*meanPeak));
		newPeakRMS[fff] = thePeakRMS;
		newPeak[fff] = thePeak;
		
		theSumFive = arraySumFive[ii][i][j][k];
		newSumFive[fff]=theSumFive;
				
		calibtree.Fill();	  	    
		std::cout <<" chamber "<<i<<" layer "<<j<<" strip "<<fff<<"  ped "<<newPed[fff]<<" RMS "<<newRMS[fff]<<" peakADC "<<newPeak[fff]<<" Peak RMS "<<newPeakRMS[fff]<<" Sum_of_four/apeak "<<newSumFive[fff]<<" size "<<std::endl;
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
	if(eventNumber !=320) {std::cout<<"Number of events not as expected "<<eventNumber<<std::endl; continue;}
	//SEND CONSTANTS TO DB
	cdb->cdb_write(test1,chamber_id,chamber_num,test2,size[myChamber]*6, newPed,    6, &ret_code);
	cdb->cdb_write(test1,chamber_id,chamber_num,test3,size[myChamber]*6, newRMS,    6, &ret_code);
	cdb->cdb_write(test1,chamber_id,chamber_num,test4,size[myChamber]*6, newPeakRMS,6, &ret_code);
	cdb->cdb_write(test1,chamber_id,chamber_num,test5,size[myChamber]*6, newSumFive,6, &ret_code);
	
	std::cout<<" Your results were sent to DB !!! "<<std::endl;
      }else{
	std::cout<<" NO data was sent!!! "<<std::endl;
      }
      }
    
    }
    calibfile.Write();
    calibfile.Close();
  }
  
 private:
  std::vector<int> newadc;
  std::vector<int> adc;
  std::string chamber_id;
  int eventNumber,evt,pedSum, strip, misMatch,fff,ret_code,NChambers,Nddu;
  int length,i_chamber,i_layer,reportedChambers,chamber_num,sector,size[CHAMBERS]; 
  float ped,pedMean,time,max,max1,aPeak,sumFive;
  float meanPedestal,meanPeak,meanPeakSquare,meanPedestalSquare,theRMS;
  float thePeak,thePeakRMS,theSumFive,thePedestal,theRSquare;
  int dmbID[CHAMBERS],crateID[CHAMBERS],l1A[CHAMBERS];    
  float arrayOfPed[DDU][CHAMBERS][LAYERS][STRIPS];
  float arrayOfPedSquare[DDU][CHAMBERS][LAYERS][STRIPS];
  float arrayPed[DDU][CHAMBERS][LAYERS][STRIPS];
  float arrayPeak[DDU][CHAMBERS][LAYERS][STRIPS];
  float arrayOfPeak[DDU][CHAMBERS][LAYERS][STRIPS];
  float arrayOfPeakSquare[DDU][CHAMBERS][LAYERS][STRIPS];
  float arraySumFive[DDU][CHAMBERS][LAYERS][STRIPS];
  float newPed[TOTALSTRIPS];
  float newRMS[TOTALSTRIPS];
  float newPeakRMS[TOTALSTRIPS];
  float newPeak[TOTALSTRIPS];
  float newSumFive[TOTALSTRIPS];
  int coinc01,coinc02,coinc12,coinc012,event;
};

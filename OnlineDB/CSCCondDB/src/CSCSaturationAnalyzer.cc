/** 
 * Analyzer for reading CSC pedestals.
 * author S. Durkin, O.Boeriu 18/03/06 
 * ripped from Jeremy's and Rick's analyzers
 *   
 */
#include <iostream>
#include <fstream>
#include <vector>
#include "string"

#include <FWCore/Framework/interface/Frameworkfwd.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigi.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigiCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "IORawData/CSCCommissioning/src/FileReaderDDU.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDDUEventData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDCCEventData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCEventData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDMBHeader.h"
#include "OnlineDB/CSCCondDB/interface/CSCSaturationAnalyzer.h"
#include "OnlineDB/CSCCondDB/interface/SaturationFit.h"

CSCSaturationAnalyzer::CSCSaturationAnalyzer(edm::ParameterSet const& conf) {
  debug = conf.getUntrackedParameter<bool>("debug",false);
  eventNumber=0,evt=0,Nddu=0;
  strip=0,misMatch=0,NChambers=0;
  i_chamber=0,i_layer=0,reportedChambers=0;
  length=1;
  aVar=0.0,bVar=0.0;

  gain_vs_charge  = TH2F("Saturation"   ,"ADC_vs_charge", 100,300,900,100,0,4000);
  gain01_vs_charge = TH2F("Saturation01","ADC_vs_charge", 100,300,900,100,0,4000);
  gain02_vs_charge = TH2F("Saturation02","ADC_vs_charge", 100,300,900,100,0,4000);
  gain03_vs_charge = TH2F("Saturation03","ADC_vs_charge", 100,300,900,100,0,4000);
  gain04_vs_charge = TH2F("Saturation04","ADC_vs_charge", 100,300,900,100,0,4000);
  gain05_vs_charge = TH2F("Saturation05","ADC_vs_charge", 100,300,900,100,0,4000);
  gain11_vs_charge = TH2F("Saturation11","ADC_vs_charge", 100,300,900,100,0,4000);
  gain12_vs_charge = TH2F("Saturation12","ADC_vs_charge", 100,300,900,100,0,4000);
  gain13_vs_charge = TH2F("Saturation13","ADC_vs_charge", 100,300,900,100,0,4000);
  gain14_vs_charge = TH2F("Saturation14","ADC_vs_charge", 100,300,900,100,0,4000);
  gain15_vs_charge = TH2F("Saturation15","ADC_vs_charge", 100,300,900,100,0,4000);
  gain21_vs_charge = TH2F("Saturation21","ADC_vs_charge", 100,300,900,100,0,4000);
  gain22_vs_charge = TH2F("Saturation22","ADC_vs_charge", 100,300,900,100,0,4000);
  gain23_vs_charge = TH2F("Saturation23","ADC_vs_charge", 100,300,900,100,0,4000);
  gain24_vs_charge = TH2F("Saturation24","ADC_vs_charge", 100,300,900,100,0,4000);
  gain25_vs_charge = TH2F("Saturation25","ADC_vs_charge", 100,300,900,100,0,4000);
  gain31_vs_charge = TH2F("Saturation31","ADC_vs_charge", 100,300,900,100,0,4000);
  gain32_vs_charge = TH2F("Saturation32","ADC_vs_charge", 100,300,900,100,0,4000);
  gain33_vs_charge = TH2F("Saturation33","ADC_vs_charge", 100,300,900,100,0,4000);
  gain34_vs_charge = TH2F("Saturation34","ADC_vs_charge", 100,300,900,100,0,4000);
  gain35_vs_charge = TH2F("Saturation35","ADC_vs_charge", 100,300,900,100,0,4000);
  gain41_vs_charge = TH2F("Saturation41","ADC_vs_charge", 100,300,900,100,0,4000);
  gain42_vs_charge = TH2F("Saturation42","ADC_vs_charge", 100,300,900,100,0,4000);
  gain43_vs_charge = TH2F("Saturation43","ADC_vs_charge", 100,300,900,100,0,4000);
  gain44_vs_charge = TH2F("Saturation44","ADC_vs_charge", 100,300,900,100,0,4000);
  gain45_vs_charge = TH2F("Saturation45","ADC_vs_charge", 100,300,900,100,0,4000);
  gain51_vs_charge = TH2F("Saturation51","ADC_vs_charge", 100,300,900,100,0,4000);
  gain52_vs_charge = TH2F("Saturation52","ADC_vs_charge", 100,300,900,100,0,4000);
  gain53_vs_charge = TH2F("Saturation53","ADC_vs_charge", 100,300,900,100,0,4000);
  gain54_vs_charge = TH2F("Saturation54","ADC_vs_charge", 100,300,900,100,0,4000);
  gain55_vs_charge = TH2F("Saturation55","ADC_vs_charge", 100,300,900,100,0,4000);
  gain61_vs_charge = TH2F("Saturation61","ADC_vs_charge", 100,300,900,100,0,4000);
  gain62_vs_charge = TH2F("Saturation62","ADC_vs_charge", 100,300,900,100,0,4000);
  gain63_vs_charge = TH2F("Saturation63","ADC_vs_charge", 100,300,900,100,0,4000);
  gain64_vs_charge = TH2F("Saturation64","ADC_vs_charge", 100,300,900,100,0,4000);
  gain65_vs_charge = TH2F("Saturation65","ADC_vs_charge", 100,300,900,100,0,4000);
  gain71_vs_charge = TH2F("Saturation71","ADC_vs_charge", 100,300,900,100,0,4000);
  gain72_vs_charge = TH2F("Saturation72","ADC_vs_charge", 100,300,900,100,0,4000);
  gain73_vs_charge = TH2F("Saturation73","ADC_vs_charge", 100,300,900,100,0,4000);
  gain74_vs_charge = TH2F("Saturation74","ADC_vs_charge", 100,300,900,100,0,4000);
  gain75_vs_charge = TH2F("Saturation75","ADC_vs_charge", 100,300,900,100,0,4000);
  gain81_vs_charge = TH2F("Saturation81","ADC_vs_charge", 100,300,900,100,0,4000);
  gain82_vs_charge = TH2F("Saturation82","ADC_vs_charge", 100,300,900,100,0,4000);
  gain83_vs_charge = TH2F("Saturation83","ADC_vs_charge", 100,300,900,100,0,4000);
  gain84_vs_charge = TH2F("Saturation84","ADC_vs_charge", 100,300,900,100,0,4000);
  gain85_vs_charge = TH2F("Saturation85","ADC_vs_charge", 100,300,900,100,0,4000);
  

  for (int i=0; i<NUMMODTEN_sat; i++){
    for (int j=0; j<CHAMBERS_sat; j++){
      for (int k=0; k<LAYERS_sat; k++){
	for (int l=0; l<STRIPS_sat; l++){
	  maxmodten[i][j][k][l] = 0.0;
	}
      }
    }
  }
  
  
  for (int i=0; i<CHAMBERS_sat; i++){
    size[i]  = 0;
  }
  
  for (int iii=0;iii<DDU_sat;iii++){
    for (int i=0; i<CHAMBERS_sat; i++){
      for (int j=0; j<LAYERS_sat; j++){
	for (int k=0; k<STRIPS_sat; k++){
	  adcMax[iii][i][j][k]            = -999.0;
	  adcMean_max[iii][i][j][k]       = -999.0;
	}
      }
    }
  }  
}

void CSCSaturationAnalyzer::analyze(edm::Event const& e, edm::EventSetup const& iSetup) {
  
  edm::Handle<CSCStripDigiCollection> strips;
  e.getByLabel("cscunpacker","MuonCSCStripDigi",strips);
  
  edm::Handle<FEDRawDataCollection> rawdata;
  e.getByType(rawdata);
  
  for (int id=FEDNumbering::getCSCFEDIds().first;
       id<=FEDNumbering::getCSCFEDIds().second; ++id){ //for each of our DCCs    
    
    /// Take a reference to this FED's data
    const FEDRawData& fedData = rawdata->FEDData(id);
    if (fedData.size()){ ///unpack data 
      
      ///get a pointer to data and pass it to constructor for unpacking
      CSCDCCEventData dccData((short unsigned int *) fedData.data()); 
      
      const std::vector<CSCDDUEventData> & dduData = dccData.dduData(); 
      
      evt++;      
      for (unsigned int iDDU=0; iDDU<dduData.size(); ++iDDU) {  ///loop over DDUs
	
	///get a reference to chamber data
	const std::vector<CSCEventData> & cscData = dduData[iDDU].cscData();
	Nddu = dduData.size();
	reportedChambers += dduData[iDDU].header().ncsc();
	NChambers = cscData.size();
	int repChambers = dduData[iDDU].header().ncsc();
	std::cout << " Reported Chambers = " << repChambers <<"   "<<NChambers<< std::endl;
	if (NChambers!=repChambers) { std::cout<< "misMatched size!!!" << std::endl; misMatch++;}
	
	for (int i_chamber=0; i_chamber<NChambers; i_chamber++) {   
	  
	  for (int i_layer = 1; i_layer <=6; ++i_layer) {
	    
	    std::vector<CSCStripDigi> digis = cscData[i_chamber].stripDigis(i_layer) ;
	    const CSCDMBHeader &thisDMBheader = cscData[i_chamber].dmbHeader();
	    
	    if (thisDMBheader.cfebAvailable()){
	      dmbID[i_chamber] = cscData[i_chamber].dmbHeader().dmbID();//get DMB ID
	      crateID[i_chamber] = cscData[i_chamber].dmbHeader().crateID();//get crate ID
	      if(crateID[i_chamber] == 255) continue;
	      
	      for (unsigned int i=0; i<digis.size(); i++){
		size[i_chamber] = digis.size();
		std::vector<int> adc = digis[i].getADCCounts();
		strip = digis[i].getStrip();
		adcMax[iDDU][i_chamber][i_layer-1][strip-1]=-99.0; 
		for(unsigned int k=0; k<adc.size(); k++){
		  float ped=(adc[0]+adc[1])/2.;
		  if(adc[k]-ped > adcMax[iDDU][i_chamber][i_layer-1][strip-1]) {
		    adcMax[iDDU][i_chamber][i_layer-1][strip-1]=adc[k]-ped;
		  }
		}
		adcMean_max[iDDU][i_chamber][i_layer-1][strip-1] += adcMax[iDDU][i_chamber][i_layer-1][strip-1]/20.;  
		
		//On the 20th event save one value
		if (evt%20 == 0 && (strip-1)%16 == (evt-1)/NUMMODTEN_sat){
		  //save 24 values from 24 settings
		  int twentyfour = int((evt-1)/20)%NUMBERPLOTTED_sat ;
		  maxmodten[twentyfour][i_chamber][i_layer-1][strip-1] = adcMean_max[iDDU][i_chamber][i_layer-1][strip-1];
		}
	      }//end digis loop
	    }//end cfeb.available loop
	  }//end layer loop
	}//end chamber loop
	
	if((evt-1)%20==0){
	  for(int iii=0;iii<DDU_sat;iii++){
	    for(int ii=0; ii<CHAMBERS_sat; ii++){
	      for(int jj=0; jj<LAYERS_sat; jj++){
		for(int kk=0; kk<STRIPS_sat; kk++){
		  adcMean_max[iii][ii][jj][kk]=0.0;
		}
	      }
	    }
	  }
	}
	
      	eventNumber++;
	edm::LogInfo ("CSCSaturationAnalyzer")  << "end of event number " << eventNumber;
      }
    }
  }
}

CSCSaturationAnalyzer::~CSCSaturationAnalyzer(){
  //  delete theOBJfun;

  //get time of Run file for DB transfer
  filein.open("../test/CSCsaturation.cfg");
  filein.ignore(1000,'\n');
  
  while(filein != NULL){
    lines++;
    getline(filein,PSet);
    
    if (lines==3){
      name=PSet;  
      cout<<name<<endl;
    }
  }
  string::size_type runNameStart = name.find("\"",0);
  string::size_type runNameEnd   = name.find("raw",0);
  string::size_type rootStart    = name.find("Saturation",0);
  int nameSize = runNameEnd+2-runNameStart;
  int myRootSize = rootStart-runNameStart+9;
  std::string myname= name.substr(runNameStart+1,nameSize);
  std::string myRootName= name.substr(runNameStart+1,myRootSize);
  std::string myRootEnd = ".root";
  std::string runFile= myRootName;
  std::string myRootFileName = runFile+myRootEnd;
  const char *myNewName=myRootFileName.c_str();
  
  struct tm* clock;			    
  struct stat attrib;			    
  stat(myname.c_str(), &attrib);          
  clock = localtime(&(attrib.st_mtime));  
  std::string myTime=asctime(clock);
    
  //DB object and map
  CSCobject *cn = new CSCobject();
  //cscmap *map = new cscmap();
  //condbon *dbon = new condbon();

  //root ntuple information
  TCalibSaturationEvt calib_evt;
  TFile calibfile(myNewName, "RECREATE");
  TTree calibtree("Calibration","Saturation");
  calibtree.Branch("EVENT", &calib_evt, "strip/I:layer/I:cham/I:id/I");

  for (int dduiter=0;dduiter<Nddu;dduiter++){
    for(int chamberiter=0; chamberiter<NChambers; chamberiter++){
      for (int cham=0;cham<NChambers;cham++){
	if (cham !=chamberiter) continue;
	
	//get chamber ID from DB mapping        
	int new_crateID = crateID[cham];
	int new_dmbID   = dmbID[cham];
	std::cout<<" Crate: "<<new_crateID<<" and DMB:  "<<new_dmbID<<std::endl;
	//map->crate_chamber(new_crateID,new_dmbID,&chamber_id,&chamber_num,&sector);
	//std::cout<<"Data is for chamber:: "<< chamber_id<<" in sector:  "<<sector<<std::endl;
	
	calib_evt.id=chamber_num;
	
	for (int layeriter=0; layeriter<LAYERS_sat; layeriter++){
	  for (int stripiter=0; stripiter<STRIPS_sat; stripiter++){
	    
	    for (int j=0; j<LAYERS_sat; j++){//layer
	      if (j != layeriter) continue;
	      
	      int layer_id=chamber_num+j+1;
	      if(sector==-100)continue;
	      cn->obj[layer_id].resize(size[cham]);

	      for (int k=0; k<size[cham]; k++){//strip
		if (k != stripiter) continue;
		
		for (int st=0;st<NUMBERPLOTTED_sat;st++){
		  myCharge[st]=0.0;
		  mySatADC[st]=0.0;
		}
		
		for(int ii=0; ii<NUMBERPLOTTED_sat; ii++){//numbers    
		  myCharge[ii] = 335.0 +(22.4*ii);
		  mySatADC[ii] = maxmodten[ii][cham][j][k];
		  gain_vs_charge.Fill(myCharge[ii],maxmodten[ii][cham][j][k]);//fill one histogram with all values for all chambers
		  //for the rest look for one strip in the middle of the CFEBs
		  if(cham==0 && k==8)  gain01_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==0 && k==24) gain02_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==0 && k==40) gain03_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==0 && k==56) gain04_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==0 && k==72) gain05_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==1 && k==8)  gain11_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==1 && k==24) gain12_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==1 && k==40) gain13_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==1 && k==56) gain14_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==1 && k==72) gain15_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==2 && k==8)  gain21_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==2 && k==24) gain22_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==2 && k==40) gain23_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==2 && k==56) gain24_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==2 && k==72) gain25_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==3 && k==8)  gain31_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==3 && k==24) gain32_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==3 && k==40) gain33_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==3 && k==56) gain34_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==3 && k==72) gain35_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==4 && k==8)  gain41_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==4 && k==24) gain42_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==4 && k==40) gain43_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==4 && k==56) gain44_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==4 && k==72) gain45_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==5 && k==8)  gain51_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==5 && k==24) gain52_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==5 && k==40) gain53_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==5 && k==56) gain54_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==5 && k==72) gain55_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==6 && k==8)  gain61_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==6 && k==24) gain62_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==6 && k==40) gain63_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==6 && k==56) gain64_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==6 && k==72) gain65_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==7 && k==8)  gain71_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==7 && k==24) gain72_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==7 && k==40) gain73_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==7 && k==56) gain74_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==7 && k==72) gain75_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==8 && k==8)  gain81_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==8 && k==24) gain82_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==8 && k==40) gain83_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==8 && k==56) gain84_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==8 && k==72) gain85_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);

		  //Use Minuit to do the fit
		  
		  calib_evt.strip     = k;
		  calib_evt.layer     = j;
		  calib_evt.cham      = cham;
		  
		  calibtree.Fill();
		  
		  cn->obj[layer_id][k].resize(3);
		}//number_plotted

		SaturationFit s(NUMBERPLOTTED_sat,myCharge,mySatADC);
		      
	      }//strip
	    }//j loop
	  }//stripiter loop
	}//layiter loop
      }//cham 
    }//chamberiter
  }//dduiter

  //send data to DB
  //dbon->cdbon_last_record("gains",&record);
  //std::cout<<"Last gains record "<<record<<" for run file "<<myname<<" saved "<<myTime<<std::endl;
  //if(debug) dbon->cdbon_write(cn,"gains",11,myTime);

  //write histograms 
  gain_vs_charge.Write();
  gain01_vs_charge.Write();
  gain02_vs_charge.Write();
  gain03_vs_charge.Write();
  gain04_vs_charge.Write();
  gain05_vs_charge.Write();
  gain11_vs_charge.Write();
  gain12_vs_charge.Write();
  gain13_vs_charge.Write();
  gain14_vs_charge.Write();
  gain15_vs_charge.Write();
  gain21_vs_charge.Write();
  gain22_vs_charge.Write();
  gain23_vs_charge.Write();
  gain24_vs_charge.Write();
  gain25_vs_charge.Write();
  gain31_vs_charge.Write();
  gain32_vs_charge.Write();
  gain33_vs_charge.Write();
  gain34_vs_charge.Write();
  gain35_vs_charge.Write();
  gain41_vs_charge.Write();
  gain42_vs_charge.Write();
  gain43_vs_charge.Write();
  gain44_vs_charge.Write();
  gain45_vs_charge.Write();
  gain51_vs_charge.Write();
  gain52_vs_charge.Write();
  gain53_vs_charge.Write();
  gain54_vs_charge.Write();
  gain55_vs_charge.Write();
  gain61_vs_charge.Write();
  gain62_vs_charge.Write();
  gain63_vs_charge.Write();
  gain64_vs_charge.Write();
  gain65_vs_charge.Write();
  gain71_vs_charge.Write();
  gain72_vs_charge.Write();
  gain73_vs_charge.Write();
  gain74_vs_charge.Write();
  gain75_vs_charge.Write();
  gain81_vs_charge.Write();
  gain82_vs_charge.Write();
  gain83_vs_charge.Write();
  gain84_vs_charge.Write();
  gain85_vs_charge.Write();

  calibfile.Write();
  calibfile.Close();
}

/** 
 * Analyzer for reading CSC ADC and injected charge for saturation.
 * author S. Durkin, O.Boeriu 16/11/06    
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
#include "DataFormats/Common/interface/Handle.h"
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

  adc_vs_charge  = TH2F("Saturation"   ,"ADC_vs_charge", 100,0,700,100,0,4000);
  adc01_vs_charge = TH2F("Saturation01","ADC_vs_charge", 100,0,700,100,0,4000);
  adc02_vs_charge = TH2F("Saturation02","ADC_vs_charge", 100,0,700,100,0,4000);
  adc03_vs_charge = TH2F("Saturation03","ADC_vs_charge", 100,0,700,100,0,4000);
  adc04_vs_charge = TH2F("Saturation04","ADC_vs_charge", 100,0,700,100,0,4000);
  adc05_vs_charge = TH2F("Saturation05","ADC_vs_charge", 100,0,700,100,0,4000);
  adc11_vs_charge = TH2F("Saturation11","ADC_vs_charge", 100,0,700,100,0,4000);
  adc12_vs_charge = TH2F("Saturation12","ADC_vs_charge", 100,0,700,100,0,4000);
  adc13_vs_charge = TH2F("Saturation13","ADC_vs_charge", 100,0,700,100,0,4000);
  adc14_vs_charge = TH2F("Saturation14","ADC_vs_charge", 100,0,700,100,0,4000);
  adc15_vs_charge = TH2F("Saturation15","ADC_vs_charge", 100,0,700,100,0,4000);
  adc21_vs_charge = TH2F("Saturation21","ADC_vs_charge", 100,0,700,100,0,4000);
  adc22_vs_charge = TH2F("Saturation22","ADC_vs_charge", 100,0,700,100,0,4000);
  adc23_vs_charge = TH2F("Saturation23","ADC_vs_charge", 100,0,700,100,0,4000);
  adc24_vs_charge = TH2F("Saturation24","ADC_vs_charge", 100,0,700,100,0,4000);
  adc25_vs_charge = TH2F("Saturation25","ADC_vs_charge", 100,0,700,100,0,4000);
  adc31_vs_charge = TH2F("Saturation31","ADC_vs_charge", 100,0,700,100,0,4000);
  adc32_vs_charge = TH2F("Saturation32","ADC_vs_charge", 100,0,700,100,0,4000);
  adc33_vs_charge = TH2F("Saturation33","ADC_vs_charge", 100,0,700,100,0,4000);
  adc34_vs_charge = TH2F("Saturation34","ADC_vs_charge", 100,0,700,100,0,4000);
  adc35_vs_charge = TH2F("Saturation35","ADC_vs_charge", 100,0,700,100,0,4000);
  adc41_vs_charge = TH2F("Saturation41","ADC_vs_charge", 100,0,700,100,0,4000);
  adc42_vs_charge = TH2F("Saturation42","ADC_vs_charge", 100,0,700,100,0,4000);
  adc43_vs_charge = TH2F("Saturation43","ADC_vs_charge", 100,0,700,100,0,4000);
  adc44_vs_charge = TH2F("Saturation44","ADC_vs_charge", 100,0,700,100,0,4000);
  adc45_vs_charge = TH2F("Saturation45","ADC_vs_charge", 100,0,700,100,0,4000);
  adc51_vs_charge = TH2F("Saturation51","ADC_vs_charge", 100,0,700,100,0,4000);
  adc52_vs_charge = TH2F("Saturation52","ADC_vs_charge", 100,0,700,100,0,4000);
  adc53_vs_charge = TH2F("Saturation53","ADC_vs_charge", 100,0,700,100,0,4000);
  adc54_vs_charge = TH2F("Saturation54","ADC_vs_charge", 100,0,700,100,0,4000);
  adc55_vs_charge = TH2F("Saturation55","ADC_vs_charge", 100,0,700,100,0,4000);
  adc61_vs_charge = TH2F("Saturation61","ADC_vs_charge", 100,0,700,100,0,4000);
  adc62_vs_charge = TH2F("Saturation62","ADC_vs_charge", 100,0,700,100,0,4000);
  adc63_vs_charge = TH2F("Saturation63","ADC_vs_charge", 100,0,700,100,0,4000);
  adc64_vs_charge = TH2F("Saturation64","ADC_vs_charge", 100,0,700,100,0,4000);
  adc65_vs_charge = TH2F("Saturation65","ADC_vs_charge", 100,0,700,100,0,4000);
  adc71_vs_charge = TH2F("Saturation71","ADC_vs_charge", 100,0,700,100,0,4000);
  adc72_vs_charge = TH2F("Saturation72","ADC_vs_charge", 100,0,700,100,0,4000);
  adc73_vs_charge = TH2F("Saturation73","ADC_vs_charge", 100,0,700,100,0,4000);
  adc74_vs_charge = TH2F("Saturation74","ADC_vs_charge", 100,0,700,100,0,4000);
  adc75_vs_charge = TH2F("Saturation75","ADC_vs_charge", 100,0,700,100,0,4000);
  adc81_vs_charge = TH2F("Saturation81","ADC_vs_charge", 100,0,700,100,0,4000);
  adc82_vs_charge = TH2F("Saturation82","ADC_vs_charge", 100,0,700,100,0,4000);
  adc83_vs_charge = TH2F("Saturation83","ADC_vs_charge", 100,0,700,100,0,4000);
  adc84_vs_charge = TH2F("Saturation84","ADC_vs_charge", 100,0,700,100,0,4000);
  adc85_vs_charge = TH2F("Saturation85","ADC_vs_charge", 100,0,700,100,0,4000);  

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
	//std::cout << " Reported Chambers = " << repChambers <<"   "<<NChambers<< std::endl;
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
    
    if (lines==2){
      name=PSet;  
      std::cout<<name<<std::endl;
    }
  }
  std::string::size_type runNameStart = name.find("\"",0);
  std::string::size_type runNameEnd   = name.find("raw",0);
  std::string::size_type rootStart    = name.find("Saturation",0);
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
  calibtree.Branch("EVENT", &calib_evt, "strip/I:layer/I:cham/I:id/I:N/F:a/F:b/F:c/F");

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
		  myCharge[ii] = 22.4 +(22.4*ii);//224(3V) to 560(5V) fC
		  mySatADC[ii] = maxmodten[ii][cham][j][k];
		  //fill one histogram with all values for all chambers
		  adc_vs_charge.Fill(myCharge[ii],maxmodten[ii][cham][j][k]);
		  //for the rest look for one strip in the middle of each CFEBs
		  if(cham==0 && k==8)  adc01_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==0 && k==24) adc02_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==0 && k==40) adc03_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==0 && k==56) adc04_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==0 && k==72) adc05_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==1 && k==8)  adc11_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==1 && k==24) adc12_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==1 && k==40) adc13_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==1 && k==56) adc14_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==1 && k==72) adc15_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==2 && k==8)  adc21_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==2 && k==24) adc22_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==2 && k==40) adc23_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==2 && k==56) adc24_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==2 && k==72) adc25_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==3 && k==8)  adc31_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==3 && k==24) adc32_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==3 && k==40) adc33_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==3 && k==56) adc34_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==3 && k==72) adc35_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==4 && k==8)  adc41_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==4 && k==24) adc42_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==4 && k==40) adc43_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==4 && k==56) adc44_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==4 && k==72) adc45_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==5 && k==8)  adc51_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==5 && k==24) adc52_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==5 && k==40) adc53_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==5 && k==56) adc54_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==5 && k==72) adc55_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==6 && k==8)  adc61_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==6 && k==24) adc62_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==6 && k==40) adc63_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==6 && k==56) adc64_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==6 && k==72) adc65_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==7 && k==8)  adc71_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==7 && k==24) adc72_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==7 && k==40) adc73_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==7 && k==56) adc74_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==7 && k==72) adc75_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==8 && k==8)  adc81_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==8 && k==24) adc82_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==8 && k==40) adc83_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==8 && k==56) adc84_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		  if(cham==8 && k==72) adc85_vs_charge.Fill(myCharge[ii] ,maxmodten[ii][cham][j][k]);
		}//number_plotted

		//Use Minuit to do the fit
		float u0_ptr=0.0, u1_ptr=0.0, u2_ptr=0.0,u3_ptr=0.0;
		SaturationFit s(NUMBERPLOTTED_sat,myCharge,mySatADC,&u0_ptr, &u1_ptr, &u2_ptr, &u3_ptr);
		//u0_ptr=N,u1_ptr=a,u2_ptr=b,u3_ptr=c
		std::cout<<"Fitresults: " <<cham<<" lay "<<j<<" strip " <<k<<" param0-3  "<< u0_ptr<<" "<<u1_ptr<<" "<<u2_ptr<<" "<<u3_ptr<<std::endl;

		calib_evt.strip = k;
		calib_evt.layer = j;
		calib_evt.cham  = cham;
		calib_evt.N     = u0_ptr;
		calib_evt.a     = u1_ptr;
		calib_evt.b     = u2_ptr;
		calib_evt.c     = u3_ptr;

		calibtree.Fill();

		//send constants to DB
		cn->obj[layer_id][k].resize(4);
		cn->obj[layer_id][k][0] = u0_ptr;
		cn->obj[layer_id][k][1] = u1_ptr;
		cn->obj[layer_id][k][2] = u2_ptr;
		cn->obj[layer_id][k][3] = u3_ptr;
	      }//strip
	    }//j loop
	  }//stripiter loop
	}//layiter loop
      }//cham 
    }//chamberiter
  }//dduiter

  //send data to DB
  //dbon->cdbon_last_record("saturation",&record);
  //std::cout<<"Last saturation record "<<record<<" for run file "<<myname<<" saved "<<myTime<<std::endl;
  //if(debug) dbon->cdbon_write(cn,"saturation",11,myTime);

  //write histograms 
  adc_vs_charge.Write();
  adc01_vs_charge.Write();
  adc02_vs_charge.Write();
  adc03_vs_charge.Write();
  adc04_vs_charge.Write();
  adc05_vs_charge.Write();
  adc11_vs_charge.Write();
  adc12_vs_charge.Write();
  adc13_vs_charge.Write();
  adc14_vs_charge.Write();
  adc15_vs_charge.Write();
  adc21_vs_charge.Write();
  adc22_vs_charge.Write();
  adc23_vs_charge.Write();
  adc24_vs_charge.Write();
  adc25_vs_charge.Write();
  adc31_vs_charge.Write();
  adc32_vs_charge.Write();
  adc33_vs_charge.Write();
  adc34_vs_charge.Write();
  adc35_vs_charge.Write();
  adc41_vs_charge.Write();
  adc42_vs_charge.Write();
  adc43_vs_charge.Write();
  adc44_vs_charge.Write();
  adc45_vs_charge.Write();
  adc51_vs_charge.Write();
  adc52_vs_charge.Write();
  adc53_vs_charge.Write();
  adc54_vs_charge.Write();
  adc55_vs_charge.Write();
  adc61_vs_charge.Write();
  adc62_vs_charge.Write();
  adc63_vs_charge.Write();
  adc64_vs_charge.Write();
  adc65_vs_charge.Write();
  adc71_vs_charge.Write();
  adc72_vs_charge.Write();
  adc73_vs_charge.Write();
  adc74_vs_charge.Write();
  adc75_vs_charge.Write();
  adc81_vs_charge.Write();
  adc82_vs_charge.Write();
  adc83_vs_charge.Write();
  adc84_vs_charge.Write();
  adc85_vs_charge.Write();

  calibfile.Write();
  calibfile.Close();
}

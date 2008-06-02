/** 
 * Analyzer for reading CSC pedestals.
 * author S. Durkin, O.Boeriu 18/03/06 
 * ripped from Jeremys and Rick's analyzers
 *   
 */
#include <iostream>
#include <fstream>
#include <vector>
#include "string"
#include <cmath>

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
#include "OnlineDB/CSCCondDB/interface/CSCGainAnalyzer.h"
#include "CondFormats/CSCObjects/interface/CSCMapItem.h"
///to be used for old mapping
//#include "OnlineDB/CSCCondDB/interface/CSCMap.h"
///for new mapping
#include "OnlineDB/CSCCondDB/interface/CSCMap1.h"

CSCGainAnalyzer::CSCGainAnalyzer(edm::ParameterSet const& conf) {
  debug = conf.getUntrackedParameter<bool>("debug",false);
  eventNumber=0,evt=0,counterzero=0,Nddu=0,flagGain=-9,flagIntercept=-9,flagRun=-9;
  strip=0,misMatch=0,myIndex=0,myNcham=-999;
  i_chamber=0,i_layer=0,reportedChambers=0;
  length=1,gainSlope=-999.0,gainIntercept=-999.0;
  
  adcCharge_ch0 = TH2F("adcCharge Cham 0","adcCharge", 100,0,300,100,0,3000);
  adcCharge_ch1 = TH2F("adcCharge Cham 1","adcCharge", 100,0,300,100,0,3000);
  adcCharge_ch2 = TH2F("adcCharge Cham 2","adcCharge", 100,0,300,100,0,3000);
  adcCharge_ch3 = TH2F("adcCharge Cham 3","adcCharge", 100,0,300,100,0,3000);
  adcCharge_ch4 = TH2F("adcCharge Cham 4","adcCharge", 100,0,300,100,0,3000);
  adcCharge_ch5 = TH2F("adcCharge Cham 5","adcCharge", 100,0,300,100,0,3000);
  adcCharge_ch6 = TH2F("adcCharge Cham 6","adcCharge", 100,0,300,100,0,3000);
  adcCharge_ch7 = TH2F("adcCharge Cham 7","adcCharge", 100,0,300,100,0,3000);
  adcCharge_ch8 = TH2F("adcCharge Cham 8","adcCharge", 100,0,300,100,0,3000);
  adcCharge_ch9 = TH2F("adcCharge Cham 9","adcCharge", 100,0,300,100,0,3000);
  adcCharge_ch10 = TH2F("adcCharge Cham 10","adcCharge", 100,0,300,100,0,3000);
  adcCharge_ch11 = TH2F("adcCharge Cham 11","adcCharge", 100,0,300,100,0,3000);
  adcCharge_ch12 = TH2F("adcCharge Cham 12","adcCharge", 100,0,300,100,0,3000);

  for (int i=0; i<NUMMODTEN_ga; i++){
    for (int j=0; j<CHAMBERS_ga; j++){
      for (int k=0; k<LAYERS_ga; k++){
	for (int l=0; l<STRIPS_ga; l++){
	  maxmodten[i][j][k][l] = 0.0;
	}
      }
    }
  }
  

  for (int i=0; i<CHAMBERS_ga; i++){
    size[i]  = 0;
  }

  for (int iii=0;iii<DDU_ga;iii++){
    for (int i=0; i<CHAMBERS_ga; i++){
      for (int j=0; j<LAYERS_ga; j++){
	for (int k=0; k<STRIPS_ga; k++){
	  adcMax[iii][i][j][k]            = -999.0;
	  adcMean_max[iii][i][j][k]       = -999.0;
	}
      }
    }
  }

  for (int i=0; i<480; i++){
    newGain[i]     =0.0;
    newIntercept[i]=0.0;
    newChi2[i]     =0.0;
  }
}

void CSCGainAnalyzer::analyze(edm::Event const& e, edm::EventSetup const& iSetup) {
  
  edm::Handle<CSCStripDigiCollection> strips;
  e.getByLabel("cscunpacker","MuonCSCStripDigi",strips);
  
  edm::Handle<FEDRawDataCollection> rawdata;
  e.getByType(rawdata);
  counterzero=counterzero+1;
  evt=(counterzero+1)/2;

  for (int id=FEDNumbering::getCSCFEDIds().first;
       id<=FEDNumbering::getCSCFEDIds().second; ++id){ //for each of our DCCs    
    
    /// Take a reference to this FED's data
    const FEDRawData& fedData = rawdata->FEDData(id);
    if (fedData.size()){ ///unpack data 
      
      ///get a pointer to data and pass it to constructor for unpacking
      CSCDCCEventData dccData((short unsigned int *) fedData.data()); 
      
      const std::vector<CSCDDUEventData> & dduData = dccData.dduData(); 
      //evt++;      
      for (unsigned int iDDU=0; iDDU<dduData.size(); ++iDDU) {  ///loop over DDUs

	///get a reference to chamber data
	const std::vector<CSCEventData> & cscData = dduData[iDDU].cscData();
	
	Nddu = dduData.size();
	reportedChambers += dduData[iDDU].header().ncsc();
	NChambers = cscData.size();
	int repChambers = dduData[iDDU].header().ncsc();
	std::cout << " Reported Chambers = " << repChambers <<"   "<<NChambers<< std::endl;
	if (NChambers!=repChambers) { std::cout<< "misMatched size!!!" << std::endl; misMatch++;}
	//if(NChambers==0) continue;
	if(NChambers > myNcham){
	  myNcham=NChambers;
	}

	float ped=0.0;

	for (int i_chamber=0; i_chamber<NChambers; i_chamber++) {   
	  
	  for (int i_layer = 1; i_layer <=6; ++i_layer) {
	    
	    std::vector<CSCStripDigi> digis = cscData[i_chamber].stripDigis(i_layer) ;
	    const CSCDMBHeader * thisDMBheader = cscData[i_chamber].dmbHeader();
	    
	    if (cscData[i_chamber].dmbHeader() && thisDMBheader->cfebAvailable()){
	      dmbID[i_chamber] = cscData[i_chamber].dmbHeader()->dmbID();//get DMB ID
	      crateID[i_chamber] = cscData[i_chamber].dmbHeader()->crateID();//get crate ID
	      if(crateID[i_chamber] == 255) continue;
	      
	      for (unsigned int i=0; i<digis.size(); i++){
		size[i_chamber] = digis.size();
		std::vector<int> adc = digis[i].getADCCounts();
		strip = digis[i].getStrip();
		adcMax[iDDU][i_chamber][i_layer-1][strip-1]=-99.0; 
		for(unsigned int k=0; k<adc.size(); k++){
		  ped=(adc[0]+adc[1])/2.;
		  if(adc[k]-ped > adcMax[iDDU][i_chamber][i_layer-1][strip-1]) {
		    adcMax[iDDU][i_chamber][i_layer-1][strip-1]=adc[k]-ped;
		  }
		}
		adcMean_max[iDDU][i_chamber][i_layer-1][strip-1] += adcMax[iDDU][i_chamber][i_layer-1][strip-1]/PULSES_ga;  
		
		//Every 25 event save
		if (evt%PULSES_ga == 0 && (strip-1)%16 == (evt-1)/NUMMODTEN_ga){
		  int pulsesNr = int((evt-1)/PULSES_ga)%NUMBERPLOTTED_ga ;
		  maxmodten[pulsesNr][i_chamber][i_layer-1][strip-1] = adcMean_max[iDDU][i_chamber][i_layer-1][strip-1];
		}
	      }//end digis loop
	    }//end cfeb.available loop
	  }//end layer loop
	}//end chamber loop
	
	//every 25 events reset
	if((evt-1)%PULSES_ga==0){
	  for(int iii=0;iii<DDU_ga;iii++){
	    for(int ii=0; ii<CHAMBERS_ga; ii++){
	      for(int jj=0; jj<LAYERS_ga; jj++){
		for(int kk=0; kk<STRIPS_ga; kk++){
		  adcMean_max[iii][ii][jj][kk]=0.0;
		}
	      }
	    }
	  }
	}
      	eventNumber++;
	edm::LogInfo ("CSCGainAnalyzer")  <<"end of event number "<<eventNumber<<" and non-zero event "<<evt;
      }
    }
  }
}


CSCGainAnalyzer::~CSCGainAnalyzer(){
  //get time of Run file for DB transfer
  filein.open("../test/CSCgain.cfg");
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
  std::string::size_type rootStart    = name.find("Gains",0);
  int nameSize = runNameEnd+2-runNameStart;
  int myRootSize = rootStart-runNameStart+8;
  std::string myname= name.substr(runNameStart+1,nameSize);
  std::string myRootName= name.substr(runNameStart+1,myRootSize);
  std::string myRootEnd = ".root";
  std::string myASCIIFileEnd = ".dat";
  std::string runFile= myRootName;
  std::string myRootFileName = runFile+myRootEnd;
  std::string myASCIIFileName= runFile+myASCIIFileEnd;
  const char *myNewName=myRootFileName.c_str();
  const char *myFileName=myASCIIFileName.c_str();
  
  struct tm* clock;			    
  struct stat attrib;			    
  stat(myname.c_str(), &attrib);          
  clock = localtime(&(attrib.st_mtime));  
  std::string myTime=asctime(clock);
  std::ofstream myGainsFile(myFileName,std::ios::out);
  std::ofstream badstripsFile1("badstrips1.dat",std::ios::out);
  std::ofstream badstripsFile2("badstrips2.dat",std::ios::out);
  
  ///old DB map
  //cscmap *map = new cscmap();
  ///new DB mapping
  CSCMapItem::MapItem mapitem;
  cscmap1 *map = new cscmap1();

  CSCobject *cn = new CSCobject();
  condbon *dbon = new condbon();

  //root ntuple information
  TCalibGainEvt calib_evt;
  TFile calibfile(myNewName, "RECREATE");
  TTree calibtree("Calibration","Gains");
  calibtree.Branch("EVENT", &calib_evt, "slope/F:intercept/F:chi2/F:strip/I:layer/I:cham/I:id/I:flagGain/I:flagIntercept/I:flagRun/I");
 
  int pointer=0;
  int bad_chan=0;
  int my_pointer=0;
  int maxBadChan=0;
  int nrBadChanCounter=0;
  int chanCounter=0;
  std::vector<int> badChanVector(9);

  for (int dduiter=0;dduiter<Nddu;dduiter++){
     for(int chamberiter=0; chamberiter<myNcham; chamberiter++){
       for (int cham=0;cham<myNcham;cham++){
	 if (cham !=chamberiter) continue;
	 int nrBadChan=0; 
	//get chamber ID from DB mapping        
	int new_crateID = crateID[cham];
	int new_dmbID   = dmbID[cham];
	int counter=0;
	std::cout<<" Crate: "<<new_crateID<<" and DMB:  "<<new_dmbID<<std::endl;
	//	myIndex=0;

	///old mapping
	//map->crate_chamber(new_crateID,new_dmbID,&chamber_id,&chamber_num,&sector,&first_strip_index,&strips_per_layer,&chamber_index);
	///new mapping
	map->cratedmb(new_crateID,new_dmbID,&mapitem);
	chamber_num=mapitem.chamberId;
	sector= mapitem.sector;
	first_strip_index=mapitem.stripIndex;
	strips_per_layer=mapitem.strips;
	chamber_index=mapitem.chamberId;
	chamber_type = mapitem.chamberLabel;
	chamberIndex = mapitem.cscIndex;

	std::cout<<"Data is for chamber:: "<<chamber_type<<"  "<<chamber_id<<" in sector:  "<<sector<<" index "<<first_strip_index<<std::endl;
	
	calib_evt.id=chamber_num;

	//badstripsFile<<"  "<<chamberIndex<<"  "<<*my_pointer<<"  "<<*totalBadChan<<std::endl;

	for (int layeriter=0; layeriter<LAYERS_ga; layeriter++){
	  for (int stripiter=0; stripiter<STRIPS_ga; stripiter++){
	      
	    for (int j=0; j<LAYERS_ga; j++){//layer
	      if (j != layeriter) continue;
	      
	      int layer_id=chamber_num+j+1;

	      //if(sector==-100)continue;
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
		//pointer=0;
		bad_chan=0;
		
		//		maxBadChan=0;
		//std::vector<int> BadChan;
		int flag1=0;
		int flag2=0;
		int flag3=0;

		
		//now start at 0.1V and step 0.25 inbetween
		float charge[NUMBERPLOTTED_ga]={11.2, 39.2, 67.2, 95.2, 123.2, 151.2, 179.2, 207.2, 235.2, 263.2, 291.2, 319.2, 347.2, 375.2, 403.2, 431.2, 459.2, 487.2, 515.2, 543.2};
		//float charge[NUMBERPLOTTED_ga]={11.2, 39.2, 67.2, 95.2, 123.2, 151.2, 179.2, 207.2, 235.2, 263.2};
		for(int ii=0; ii<FITNUMBERS_ga; ii++){//numbers    
		  sumOfX  += charge[ii];
		  sumOfY  += maxmodten[ii][cham][j][k];
		  sumOfXY += (charge[ii]*maxmodten[ii][cham][j][k]);
		  sumx2   += (charge[ii]*charge[ii]);
		  myCharge[ii] = 11.2 +(28.0*ii);
		  if(cham==0) adcCharge_ch0.Fill(myCharge[ii],maxmodten[ii][cham][j][k]);
		  if(cham==1) adcCharge_ch1.Fill(myCharge[ii],maxmodten[ii][cham][j][k]);
		  if(cham==2) adcCharge_ch2.Fill(myCharge[ii],maxmodten[ii][cham][j][k]);
		  if(cham==3) adcCharge_ch3.Fill(myCharge[ii],maxmodten[ii][cham][j][k]);
		  if(cham==4) adcCharge_ch4.Fill(myCharge[ii],maxmodten[ii][cham][j][k]);
		  if(cham==5) adcCharge_ch5.Fill(myCharge[ii],maxmodten[ii][cham][j][k]);
		  if(cham==6) adcCharge_ch6.Fill(myCharge[ii],maxmodten[ii][cham][j][k]);
		  if(cham==7) adcCharge_ch7.Fill(myCharge[ii],maxmodten[ii][cham][j][k]);
		  if(cham==8) adcCharge_ch8.Fill(myCharge[ii],maxmodten[ii][cham][j][k]);
		  if(cham==9) adcCharge_ch9.Fill(myCharge[ii],maxmodten[ii][cham][j][k]);
		  if(cham==10) adcCharge_ch10.Fill(myCharge[ii],maxmodten[ii][cham][j][k]);
		  if(cham==11) adcCharge_ch11.Fill(myCharge[ii],maxmodten[ii][cham][j][k]);
		  if(cham==12) adcCharge_ch12.Fill(myCharge[ii],maxmodten[ii][cham][j][k]);
		}
	
		//Fit parameters for straight line
		gainSlope     = ((FITNUMBERS_ga*sumOfXY) - (sumOfX * sumOfY))/((FITNUMBERS_ga*sumx2) - (sumOfX*sumOfX));//k
		gainIntercept = ((sumOfY*sumx2)-(sumOfX*sumOfXY))/((FITNUMBERS_ga*sumx2)-(sumOfX*sumOfX));//m
		
		//if (gainSlope <3.0)  gainSlope = 0.0;
		if (isnan(gainSlope)) gainSlope = 0.0;
		
		for(int ii=0; ii<FITNUMBERS_ga; ii++){
		  chi2  += (maxmodten[ii][cham][j][k]-(gainIntercept+(gainSlope*charge[ii])))*(maxmodten[ii][cham][j][k]-(gainIntercept+(gainSlope*charge[ii])))/(FITNUMBERS_ga*FITNUMBERS_ga);
		}
			      
		if (gainSlope>5.0 && gainSlope<10.0) flagGain=1; // ok
		if (gainSlope<5.0)                   flagGain=2; // warning fit fails
		if (gainSlope>10.0)                  flagGain=3; // warning fit fails

		if (gainIntercept> -40. && gainIntercept<15.)  flagIntercept = 1 ;
		if (gainIntercept< -40.)                       flagIntercept = 2 ;
		if (gainIntercept> 15.)                        flagIntercept = 3 ;  

		//dump values to ASCII file
		counter++; 
		myIndex = first_strip_index+counter-1;
		if (size[cham] != strips_per_layer) flagRun = 1; //bad run
		if (size[cham] == strips_per_layer) flagRun = 0; //good run 
		if (counter>size[cham]*LAYERS_ga) counter=0;

		//BadStripChannels!!!!
		myGainsFile <<"  "<<myIndex-1<<"  "<<gainSlope<<"  "<<gainIntercept<<"  "<<chi2<<std::endl;

		if(gainSlope<4.0){
		  gainSlope=0.0;
		  nrBadChan++;
		  nrBadChanCounter++;
		  flag1=1;  
		  //dead channel
		  badstripsFile2<<"  "<<j+1<<"  "<<k+1<<"  "<<flag1<<"  "<<flag2<<"  "<<flag3<<std::endl;
		}
		if(gainSlope>9.0 && gainSlope<10.0){
		  flag2=2;
		}//noisy channel
		if(gainSlope>10.0){
		  flag3=3; 
		}// hot channel
		 
		maxBadChan=nrBadChan;
		chanCounter=nrBadChanCounter;

		  //badstripsFile<<"  "<<chamberIndex<<"  "<<pointer<<"  "<<bad_chan<<"  "<<j<<"  "<<k<<"  "<<flag1<<"  "<<flag2<<"  "<<flag3<<std::endl;

		calib_evt.slope     = gainSlope;
		calib_evt.intercept = gainIntercept;
		calib_evt.chi2      = chi2;
		calib_evt.strip     = k;
		calib_evt.layer     = j;
		calib_evt.cham      = cham;
		calib_evt.flagGain  = flagGain;
		calib_evt.flagIntercept  = flagIntercept;
		calib_evt.flagRun   = flagRun;
		
		calibtree.Fill();
		
		cn->obj[layer_id][k].resize(3);
		cn->obj[layer_id][k][0] = gainSlope;
		cn->obj[layer_id][k][1] = gainIntercept;
		cn->obj[layer_id][k][2] = chi2;
	      }//k loop
	    }//j loop
	  }//stripiter loop
	}//layiter loop
	int firstPointer=0;
	badChanVector[cham]=chanCounter;
	//my_pointer = &chanCounter;
	if (cham==0) my_pointer=firstPointer;
	if (cham>0)  my_pointer=badChanVector[cham-1];
	pointer = my_pointer;
	if (maxBadChan !=0) badstripsFile1<<"  "<<chamberIndex<<"  "<<pointer+1<<"  "<<maxBadChan<<std::endl;
      }//cham 
    }//chamberiter
  }//dduiter
    
  //send data to DB
  dbon->cdbon_last_record("gains",&record);
  std::cout<<"Last gains record "<<record<<" for run file "<<myname<<" saved "<<myTime<<std::endl;
  if(debug) dbon->cdbon_write(cn,"gains",12,3498,myTime);
  adcCharge_ch0.Write();
  adcCharge_ch1.Write();
  adcCharge_ch2.Write();
  adcCharge_ch3.Write();
  adcCharge_ch4.Write();
  adcCharge_ch5.Write();
  adcCharge_ch6.Write();
  adcCharge_ch7.Write();
  adcCharge_ch8.Write();
  adcCharge_ch9.Write();
  adcCharge_ch10.Write();
  adcCharge_ch11.Write();
  adcCharge_ch12.Write();

  calibfile.Write();
  calibfile.Close();
  
}

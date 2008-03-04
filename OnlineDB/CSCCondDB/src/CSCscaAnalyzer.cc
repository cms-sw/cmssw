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
#include "EventFilter/CSCRawToDigi/interface/CSCCFEBData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCCFEBTimeSlice.h"
#include "OnlineDB/CSCCondDB/interface/CSCscaAnalyzer.h"

CSCscaAnalyzer::CSCscaAnalyzer(edm::ParameterSet const& conf) {
  debug = conf.getUntrackedParameter<bool>("debug",false);
  eventNumber=0,evt=0, Nddu=0;
  strip=0,misMatch=0,maxDDU=0,maxStrip=-999;
  chamber=0,layer=0,reportedChambers=0;
  length=1,myevt=0,flag=-9,myIndex=0;
  pedMean=0.0,NChambers=0,myNcham=0,counter=0;

  for (int i=0;i<DDU_sca;i++){
    for (int j=0;j<CHAMBERS_sca;j++){
      for (int k=0;k<LAYERS_sca;k++){
	for (int l=0;l<STRIPS_sca;l++){
	  scaNr[i][j][k][l]=0;
	}
      }
    }
  }

  for (int i=0;i<DDU_sca;i++){
    for (int j=0;j<CHAMBERS_sca;j++){
      for (int k=0;k<LAYERS_sca;k++){
	for (int l=0;l<STRIPS_sca;l++){
	  for (int m=0;m<Number_sca;m++){
	    value_adc[i][j][k][l][m]=0;
	    value_adc_mean[i][j][k][l][m]=0.0;
	    count_adc_mean[i][j][k][l][m]=0;
	    div[i][j][k][l][m]=0.0;
	  }
	}
      }
    }
  }

  //definition of histograms....

  
}

void CSCscaAnalyzer::analyze(edm::Event const& e, edm::EventSetup const& iSetup) {
 
  edm::Handle<CSCStripDigiCollection> strips;
  e.getByLabel("cscunpacker","MuonCSCStripDigi",strips);
  edm::Handle<FEDRawDataCollection> rawdata;
  e.getByType(rawdata); //before 0_7_0_pre4 use getByLabel("DaqSource", rawdata)
  //myevt=e.id().event();
  //counterzero=counterzero+1;
  //evt=(counterzero+1)/2;
  int conv_blk[16]={0,1,2,3,4,5,6,7,8,0,9,0,10,0,11,0};

  /*  
  if (NChambers !=0){
    evt++;
  }
  */

  for (int id=FEDNumbering::getCSCFEDIds().first;
       id<=FEDNumbering::getCSCFEDIds().second; ++id){ //for each of our DCCs
    
    /// Take a reference to this FED's data
    const FEDRawData& fedData = rawdata->FEDData(id);
    if (fedData.size()){ ///unpack data 
      
      ///get a pointer to data and pass it to constructor for unpacking
      CSCDCCEventData dccData((short unsigned int *) fedData.data()); 
      
      const std::vector<CSCDDUEventData> & dduData = dccData.dduData(); 
      
      //evt++;      
      for (unsigned int iDDU=0; iDDU<dduData.size(); ++iDDU) { 
	if(iDDU > maxDDU){
	  maxDDU=iDDU;
	}

	///get a reference to chamber data
	const std::vector<CSCEventData> & cscData = dduData[iDDU].cscData();
	//exclude empty events with no DMB/CFEB data
	if(dduData[iDDU].cscData().size()==0) continue;
	if(dduData[iDDU].cscData().size() !=0) evt++;
	
	Nddu = dduData.size();
	reportedChambers += dduData[iDDU].header().ncsc();
	NChambers = cscData.size();
	int repChambers = dduData[iDDU].header().ncsc();
	std::cout << " Reported Chambers = " << repChambers <<"   "<<NChambers<< std::endl;
	if (NChambers!=repChambers) { std::cout<< "misMatched size!!!" << std::endl; misMatch++; continue;}
	if(NChambers > myNcham){
	  myNcham=NChambers;
	}
	if(NChambers !=0) evt++;

	for (int chamber = 0; chamber < NChambers; chamber++){
	  const CSCDMBHeader &thisDMBheader = cscData[chamber].dmbHeader();
	  
	  if (thisDMBheader.cfebAvailable()){
	    dmbID[chamber] = cscData[chamber].dmbHeader().dmbID();
	    crateID[chamber] = cscData[chamber].dmbHeader().crateID();
	    if(crateID[chamber] == 255) continue;

	    for (int icfeb=0; icfeb<5;icfeb++) {//loop over cfebs in a given chamber
	      CSCCFEBData * mycfebData =  cscData[chamber].cfebData(icfeb);
	      if (!mycfebData) continue;

	      for (unsigned int layer = 1; layer <= 6; layer++){
		std::vector<CSCStripDigi> digis = cscData[chamber].stripDigis(layer) ;		
	
                int blk_strt=0;
		for (int itime=0;itime<8;itime++){
		  const CSCCFEBTimeSlice * mytimeSlice =  mycfebData->timeSlice(itime);
		  if (!mytimeSlice)continue;
		  
		  scaBlock = mytimeSlice->scaControllerWord(layer).sca_blk;
		  trigTime = mytimeSlice->scaControllerWord(layer).trig_time;
		  lctPhase = mytimeSlice->scaControllerWord(layer).lct_phase;
		  int tmp=1;
		  for(power=0;power<8;power++){if(trigTime==tmp) lctPhase=power; tmp=tmp*2;}
		  if(trigTime!=0){
                     cap = lctPhase+itime;
                     blk_strt=itime;
                  }else{
		    cap=itime-blk_strt-1;
                  }
		  scaNumber=8*conv_blk[scaBlock]+cap;
		  // printf(" -> itime %d scaBlock %d trigTime %d cap %d scaNumber %d \n",itime,scaBlock,trigTime,cap,scaNumber);
		  
		  for (unsigned int i=0; i<digis.size(); i++){
		    int strip = digis[i].getStrip();
		    if (strip > maxStrip){
		      maxStrip=strip;
		    }
		    adc = digis[i].getADCCounts();
		    scaNr[iDDU][chamber][layer][strip] = scaNumber;
		    std::cout<<"strip "<<strip<<"  "<<layer<<"  "<<chamber<<"  "<<iDDU<<" SCA nr "<<scaNumber<<std::endl;
		    /*
		    if(strip>=icfeb*16+1 && strip<=icfeb*16+16){
		      value_adc[iDDU][chamber][layer][strip][scaNumber] = adc[itime];
		      scaNr[iDDU][chamber][layer][strip] = scaNumber;
		    }
		    */

		    value_adc_mean[iDDU][chamber][layer][strip][scaNumber] += adc[itime];
		    count_adc_mean[iDDU][chamber][layer][strip][scaNumber] +=1;
		    
		    div[iDDU][chamber][layer][strip][scaNumber]=value_adc_mean[iDDU][chamber][layer][strip][scaNumber]/count_adc_mean[iDDU][chamber][layer][strip][scaNumber];

		    //std::cout<<"ADC mean "<<value_adc_mean[iDDU][chamber][layer][strip][scaNumber] <<"  "<<count_adc_mean[iDDU][chamber][layer][strip][scaNumber]<<std::endl;
		  }
		}//8 timeslice
	      }//layer
	    }//CFEBs
	  }//CFEB available
	}//chamber
      
	//???????????
	/*
	if((evt-1)%20==0){
	  for(int iii=0;iii<DDU_sca;iii++){
	    for(int ii=0; ii<CHAMBERS_sca; ii++){
	      for(int jj=0; jj<LAYERS_sca; jj++){
		for(int kk=0; kk<STRIPS_sca; kk++){
		  for (int m=0;m<Number_sca;m++){
		    value_adc_mean[iii][ii][jj][kk][m]=0.0;
		  }
		}
	      }
	    }
	  }
	}
	*/

	eventNumber++;
	edm::LogInfo ("CSCscaAnalyzer")  << "end of event number " << eventNumber<<" and non-zero event "<<evt;
	
      }
    }
  }
  //  loop over ddu,cham,layer,strip,scanr!!!
  /*
  for (unsigned int iDDU=0; iDDU<maxDDU; ++iDDU) { 
    for (int chamber = 0; chamber <myNcham; chamber++){
      for (unsigned int layer = 1; layer <= 6; layer++){
	for (int strip=0; strip<maxStrip; strip++){
	  for (int scaNumber=0; scaNumber<96; scaNumber++){
	    div[iDDU][chamber][layer][strip][scaNumber]=value_adc_mean[iDDU][chamber][layer][strip][scaNumber]/count_adc_mean[iDDU][chamber][layer][strip][scaNumber];
	    std::cout<<"division "<<div[iDDU][chamber][layer][strip][scaNumber]<<std::endl;
	  }
	}
      }
    }
  }
  */
}

CSCscaAnalyzer::~CSCscaAnalyzer(){
  
  //get time of Run file for DB transfer
  filein.open("../test/CSCsca.cfg");
  filein.ignore(1000,'\n');
  
  while(filein != NULL){
    lines++;
    getline(filein,PSet);
    
    if (lines==2){
      name=PSet;  
    }
  }
  
  //get name of run file from .cfg and name root output after that
  std::string::size_type runNameStart = name.find("\"",0);
  std::string::size_type runNameEnd   = name.find("raw",0);
  std::string::size_type rootStart    = name.find("SCAPed",0);
  int nameSize = runNameEnd+3-runNameStart;
  int myRootSize = rootStart-runNameStart+8;
  std::string myname= name.substr(runNameStart+1,nameSize);
  std::string myRootName= name.substr(runNameStart+1,myRootSize);
  std::string myRootEnd = "_sca.root";
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
  std::ofstream mySCAfile(myFileName,std::ios::out);

  //root ntuple
  TCalibSCAEvt calib_evt;
  TFile calibfile(myNewName, "RECREATE");
  TTree calibtree("Calibration","SCA");
  calibtree.Branch("EVENT", &calib_evt, "strip/I:layer/I:cham/I:ddu/I:scaMeanVal/F:id/I:scanumber/I");
  
  //DB object and map
  //CSCobject *cn = new CSCobject();
  //CSCobject *cn1 = new CSCobject();
  CSCMapItem::MapItem mapitem;
  cscmap1 *map = new cscmap1();
  //condbon *dbon = new condbon();

  for (int dduiter=0;dduiter<Nddu;dduiter++){ 
    for (int cham=0;cham<myNcham;cham++){ 
      
      //get chamber ID from DB mapping
      int new_crateID = crateID[cham];
      int new_dmbID   = dmbID[cham];
      
      std::cout<<" Crate: "<<new_crateID<<" and DMB:  "<<new_dmbID<<std::endl;
      //map->crate_chamber(new_crateID,new_dmbID,&chamber_id,&chamber_num,&sector,&first_strip_index,&strips_per_layer,&chamber_index);
      //new mapping
      map->cratedmb(new_crateID,new_dmbID,&mapitem);
      chamber_num=mapitem.chamberId;
      sector= mapitem.sector;
      first_strip_index=mapitem.stripIndex;
      strips_per_layer=mapitem.strips;
      chamber_index=mapitem.chamberId;
      chamber_type = mapitem.chamberLabel;

      std::cout<<"Data is for chamber:: "<< chamber_type<<"  "<<chamber_id<<" in sector:  "<<sector<<" index "<<first_strip_index<<std::endl;
      
      calib_evt.id = chamber_num;

      for (int layeriter=0; layeriter<LAYERS_sca; layeriter++){
	//int layer_id=chamber_num+layeriter+1;

	for (int stripiter=0; stripiter<STRIPS_sca; stripiter++){
	  for (int k=0;k<Number_sca;k++){
	    my_scaValue = value_adc[dduiter][cham][layeriter][stripiter][k];
	    //my_scaValueMean = value_adc_mean[dduiter][cham][layeriter][stripiter][k];
	    my_scaValueMean = div[dduiter][cham][layeriter][stripiter][k];
	    mySCAfile<<"  "<<myIndex-1<<"  "<<my_scaValueMean<<std::endl;
	    counter++;
	    myIndex = first_strip_index+counter-1;
	    //std::cout<<"Ch "<<cham<<" Layer "<<layeriter<<" strip "<<stripiter<<" sca_nr "<<k<<" Mean ADC "<<my_scaValueMean <<std::endl;
	    calib_evt.strip=stripiter;
	    calib_evt.layer=layeriter;
	    calib_evt.cham=cham;
	    calib_evt.ddu=dduiter;
	    calib_evt.scaMeanVal=my_scaValueMean;
	    calib_evt.scanumber=scaNr[dduiter][cham][layeriter][stripiter];
    
	    calibtree.Fill();
	  }
	}
      }
    }
  }
  calibfile.Write();    
  calibfile.Close(); 
}

/** 
 * Analyzer for reading CSC bin by bin ADC information for noise matrix.
 * author S. Durkin, O.Boeriu 30/11/06 
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
#include "OnlineDB/CSCCondDB/interface/CSCNoiseMatrixAnalyzer.h"

CSCNoiseMatrixAnalyzer::CSCNoiseMatrixAnalyzer(edm::ParameterSet const& conf) {
  debug = conf.getUntrackedParameter<bool>("debug",false);
  eventNumber=0,evt=0,NChambers=0,Nddu=0;
  strip=0,misMatch=0;
  i_chamber=0,i_layer=0,reportedChambers=0;
  length=1,flagMatrix=-9;
  for(int k=0;k<CHAMBERS_ma;k++) cam[k].zero();

  for (int i=0;i<480;i++){
    newMatrix1[i] = 0.0;
    newMatrix2[i] = 0.0;
    newMatrix3[i] = 0.0;
    newMatrix4[i] = 0.0;
    newMatrix5[i] = 0.0;
    newMatrix6[i] = 0.0;
    newMatrix7[i] = 0.0;
    newMatrix8[i] = 0.0;
    newMatrix9[i] = 0.0;
    newMatrix10[i]= 0.0;
    newMatrix11[i]= 0.0;
    newMatrix12[i]= 0.0;
  
  }

  for (int i=0; i< CHAMBERS_ma; i++){
    size[i]=0;
  }
}

void CSCNoiseMatrixAnalyzer::analyze(edm::Event const& e, edm::EventSetup const& iSetup) {
  
  // These declarations create handles to the types of records that you want
  // to retrieve from event "e".
  //
  edm::Handle<CSCStripDigiCollection> strips;
  
  // Pass the handle to the method "getByType", which is used to retrieve
  // one and only one instance of the type in question out of event "e". If
  // zero or more than one instance exists in the event an exception is thrown.
  //
  e.getByLabel("cscunpacker","MuonCSCStripDigi",strips);
  
  edm::Handle<FEDRawDataCollection> rawdata;
  e.getByType(rawdata);

  for (int id=FEDNumbering::getCSCFEDIds().first;
       id<=FEDNumbering::getCSCFEDIds().second; ++id){ //for each of our DCCs
    
    evt++;      
    /// Take a reference to this FED's data
    const FEDRawData& fedData = rawdata->FEDData(id);
    if (fedData.size()){ ///unpack data 
      
      ///get a pointer to data and pass it to constructor for unpacking
      CSCDCCEventData dccData((short unsigned int *) fedData.data()); 
      
      const std::vector<CSCDDUEventData> & dduData = dccData.dduData(); 
         
      for (unsigned int iDDU=0; iDDU<dduData.size(); ++iDDU) { 
	
	///get a reference to chamber data
	const std::vector<CSCEventData> & cscData = dduData[iDDU].cscData();
	Nddu=dduData.size();
	reportedChambers += dduData[iDDU].header().ncsc();
	NChambers = cscData.size();
	int repChambers = dduData[iDDU].header().ncsc();
	std::cout << " Reported Chambers = " << repChambers <<"   "<<NChambers<< std::endl;
	if (NChambers!=repChambers) { std::cout<< "misMatched size!!!" << std::endl; misMatch++;}

	for (int i_chamber=0; i_chamber<NChambers; i_chamber++) { 
	  
	  for(int i_layer = 1; i_layer <= LAYERS_ma; ++i_layer) {
	    std::vector<CSCStripDigi> digis = cscData[i_chamber].stripDigis(i_layer) ;
	    const CSCDMBHeader &thisDMBheader = cscData[i_chamber].dmbHeader();
	    
	    if (thisDMBheader.cfebAvailable()){
	      dmbID[i_chamber]   = cscData[i_chamber].dmbHeader().dmbID(); 
	      crateID[i_chamber] = cscData[i_chamber].dmbHeader().crateID();
	      if(crateID[i_chamber] == 255) continue; 

	      for (unsigned int i=0; i<digis.size(); i++){
		size[i_chamber]=digis.size();
		int strip = digis[i].getStrip();
		adc = digis[i].getADCCounts();
		int tadc[8];
		for(unsigned int j=0;j<adc.size();j++)tadc[j]=adc[j];
		cam[i_chamber].add(i_layer-1,strip-1,tadc);
	      }
	    }
	  }
	}
	tmp=corrmat; 
		
	eventNumber++;
	edm::LogInfo ("CSCNoiseMatrixAnalyzer")  << "end of event number " << eventNumber;
      }
    }
  }
}


CSCNoiseMatrixAnalyzer::~CSCNoiseMatrixAnalyzer(){
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
  string::size_type runNameStart = name.find("\"",0);
  string::size_type runNameEnd   = name.find("raw",0);
  string::size_type rootStart    = name.find("PulseDAC",0);
  int nameSize = runNameEnd+2-runNameStart;
  int myRootSize = rootStart-runNameStart+11;
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
  cscmap *map = new cscmap();
  condbon *dbon = new condbon();
  
  //root ntuple
  TCalibNoiseMatrixEvt calib_evt;
  TFile calibfile(myNewName, "RECREATE");
  TTree calibtree("Calibration","NoiseMatrix");
  calibtree.Branch("EVENT", &calib_evt, "elem[12]/F:strip/I:layer/I:cham/I:id/I:flagMatrix/I");
  
  //for (int myDDU; myDDU<Nddu; myDDU++){
  for (int i=0; i<NChambers; i++){
    
    //get chamber ID from DB mapping        
    int new_crateID = crateID[i];
    int new_dmbID   = dmbID[i];
    std::cout<<" Crate: "<<new_crateID<<" and DMB:  "<<new_dmbID<<std::endl;
    map->crate_chamber(new_crateID,new_dmbID,&chamber_id,&chamber_num,&sector);
    std::cout<<"Data is for chamber:: "<< chamber_id<<" in sector:  "<<sector<<std::endl;
       
    calib_evt.id=chamber_num;
    for (int j=0; j<LAYERS_ma; j++){
      int layer_id=chamber_num+j+1;
      if(sector==-100)continue;
      cn->obj[layer_id].resize(size[i]);
      
      for (int k=0; k<size[i]; k++){
	for (int max=0; max<12;max++){
	  tmp=cam[i].autocorrmat(j,k);
	  
	  if (tmp[max]>3.0 && tmp[max]<100.0) flagMatrix = 1; // ok
	  if (tmp[max]>50.0)                  flagMatrix = 2; // warning too high
	  if (tmp[max]<3.0)                   flagMatrix = 3; // warning too low

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
	  calib_evt.flagMatrix = flagMatrix;
	  
	  calibtree.Fill();
	  
	  std::cout<<"Chamber "<<i<<" Layer "<<j<<" strip "<<k<<" Matrix elements "<<tmp[max]<<std::endl;
	  
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
  dbon->cdbon_last_record("noisematrix",&record);
  std::cout<<"record "<<record<<" for run file "<<myname<<" saved "<<myTime<<std::endl;
  if(debug) dbon->cdbon_write(cn,"noisematrix",12,myTime);
  std::cout<<"record "<<record<<" for run file "<<myname<<" saved "<<myTime<<std::endl;
  
  calibfile.Write();
  calibfile.Close();
}

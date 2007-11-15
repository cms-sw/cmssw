/** 
 * Analyzer for reading mean ADC from CFEB connectivity .
 * author O.Boeriu 14/07/06 
 * ripped from Jeremy's and Rick's analyzers
 *   
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

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
#include "OnlineDB/CSCCondDB/interface/CSCCFEBConnectivityAnalyzer.h"

CSCCFEBConnectivityAnalyzer::CSCCFEBConnectivityAnalyzer(edm::ParameterSet const& conf) {

  debug = conf.getUntrackedParameter<bool>("debug",false);
  eventNumber = 0,strip=0,flagConnect=-9;
  evt = 0,Nddu=0,misMatch=0,myIndex=0,myNcham=-999;
  chamber=0,layer=0,reportedChambers =0;
  length = 1, NChambers=0;
   
  //initialize arrays
  for (int ii=0;ii<DDU_con;ii++){
    for (int i=0; i<CHAMBERS_con; i++){
      for (int j=0; j<LAYERS_con; j++){
	for (int k=0; k<STRIPS_con;k++){
	  adcMin[ii][i][j][k]    = 9999999.0;
	  adcMax[ii][i][j][k]    = -9999999.0;
	  adcMean_max[ii][i][j][k]=0.0;
	  adcMean_min[ii][i][j][k]=0.0;
	  diff[ii][i][j][k]      = 0.0;
	}
      }
    }
  }
}

void CSCCFEBConnectivityAnalyzer::analyze(edm::Event const& e, edm::EventSetup const& iSetup) {
  
  // These declarations create handles to the types of records that you want
  // to retrieve from event "e".
  //
  edm::Handle<CSCStripDigiCollection> strips;
  
  e.getByLabel("cscunpacker","MuonCSCStripDigi",strips);
  
  edm::Handle<FEDRawDataCollection> rawdata;
  e.getByType(rawdata);
  //event =e.id().event();
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
	Nddu=dduData.size();
	reportedChambers += dduData[iDDU].header().ncsc();
	NChambers = cscData.size();
	int repChambers = dduData[iDDU].header().ncsc();
	std::cout << " Reported Chambers = " << repChambers <<"   "<<NChambers<< std::endl;
	if (NChambers!=repChambers) {std::cout<< "misMatched size!!!" << std::endl; misMatch++;continue;}
	if(NChambers > myNcham){
	  myNcham=NChambers;
	}
	
	if (NChambers !=0){
	  evt++;  
	}

	for (chamber=0; chamber<NChambers; chamber++) {//loop over all DMBs  
	  
	  for(layer = 1; layer <= 6; ++layer) {//loop over all layers in chambers
	    	    
	    std::vector<CSCStripDigi> digis = cscData[chamber].stripDigis(layer) ;
	    const CSCDMBHeader &thisDMBheader = cscData[chamber].dmbHeader();
	     
	    if (thisDMBheader.cfebAvailable()){//check that CFEB data exists
	      
	      dmbID[chamber]   = cscData[chamber].dmbHeader().dmbID(); //get DMB ID
	      crateID[chamber] = cscData[chamber].dmbHeader().crateID(); //get crate ID
	      if(crateID[chamber] == 255) continue; //255 doesn't exist
	      
	      for (unsigned int i=0; i<digis.size(); i++){
		size[chamber] = digis.size();
		strip = digis[i].getStrip();
		std::vector<int> adc = digis[i].getADCCounts();
		
		for(unsigned int k=0; k<adc.size(); k++){
		  
		  if(adc[k] > adcMax[iDDU][chamber][layer-1][strip-1]) {
		    adcMax[iDDU][chamber][layer-1][strip-1]= adc[k];
		  }
		  
		  if(adc[k] < adcMin[iDDU][chamber][layer-1][strip-1]){
		    adcMin[iDDU][chamber][layer-1][strip-1]= adc[k];
		  }
		  
		}//end timebins loop

		adcMean_max[iDDU][chamber][layer-1][strip-1] += adcMax[iDDU][chamber][layer-1][strip-1]/25.;
		adcMean_min[iDDU][chamber][layer-1][strip-1] += adcMin[iDDU][chamber][layer-1][strip-1]/25.;
	      
	      }//end digis size
	    }//end if cfeb.available 
	  }//end loop over layers
        }//end loop over chambers
	
	if((evt-1)%25==0){
	  for(int iii=0;iii<DDU_con;iii++){
	    for(int ii=0; ii<CHAMBERS_con; ii++){
	      for(int jj=0; jj<LAYERS_con; jj++){
		for(int kk=0; kk<STRIPS_con; kk++){
		  adcMean_max[iii][ii][jj][kk]=0.0;
		  adcMean_min[iii][ii][jj][kk]=0.0;
		}
	      }
	    }
	  }
	}
	
	eventNumber++;
	edm::LogInfo ("CSCCFEBConnectivityAnalyzer")  << "end of event number " << eventNumber;
	
      }
    }
  }
}

CSCCFEBConnectivityAnalyzer::~CSCCFEBConnectivityAnalyzer(){

  //get time of Run file for DB transfer
  filein.open("../test/CSCCFEBconnect.cfg");
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
  std::string::size_type rootStart    = name.find("Cross",0);
  int nameSize = runNameEnd+2-runNameStart;
  int myRootSize = rootStart-runNameStart+8;
  std::string myname= name.substr(runNameStart+1,nameSize);
  std::string myRootName= name.substr(runNameStart+1,myRootSize);
  std::string myRootEnd = "Connectivity.root";
  std::string myASCIIFileEnd = "Connectivity.dat";
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
  std::ofstream myConnectFile(myFileName,std::ios::out);

  //old DB map
  //cscmap *map = new cscmap();
  //condbon *dbon = new condbon();
  CSCMapItem::MapItem mapitem;
  cscmap1 *map = new cscmap1(); 
  CSCobject *cn = new CSCobject();
  
  //root ntuple
  TCalibCFEBConnectEvt calib_evt;
  TFile calibfile(myNewName, "RECREATE");
  TTree calibtree("Calibration","Connectivity");
  calibtree.Branch("EVENT", &calib_evt, "strip/I:layer/I:cham/I:ddu/I:adcMax/F:adcMin/F:diff/F:RMS/F:id:flagConnect/I");
  
  for (int iii=0; iii<Nddu; iii++){
    for (int i=0; i<myNcham; i++){
      theRMS      =0.0;
      my_diffSquare=0.0;
      
      //get chamber ID from DB mapping        
      int new_crateID = crateID[i];
      int new_dmbID   = dmbID[i];
      int counter=0;
      std::cout<<" Crate: "<<new_crateID<<" and DMB:  "<<new_dmbID<<std::endl;
      //old map call
      // map->crate_chamber(new_crateID,new_dmbID,&chamber_id,&chamber_num,&sector,&first_strip_index,&strips_per_layer,&chamber_index);
      ///new mapping
	map->cratedmb(new_crateID,new_dmbID,&mapitem);
	chamber_num=mapitem.chamberId;
	sector= mapitem.sector;
	first_strip_index=mapitem.stripIndex;
	strips_per_layer=mapitem.strips;
	chamber_index=mapitem.chamberId;
	chamber_type = mapitem.chamberLabel;

	std::cout<<"Data is for chamber:: "<< chamber_type<<"  "<<chamber_id<<" in sector:  "<<" index "<<first_strip_index<<sector<<std::endl;
      
      calib_evt.id = chamber_num;

      for (int j=0; j<LAYERS_con; j++){
	int layer_id=chamber_num+j+1;
	if(sector==-100)continue;
	cn->obj[layer_id].resize(size[i]);
	for (int k=0; k<size[i]; k++){
	  
	  my_diff =  adcMean_max[iii][i][j][k]- adcMean_min[iii][i][j][k];
	  my_diffSquare = my_diff*my_diff;
	  //std::cout<<"Chamber "<<i<<" Layer "<<j<<" Strip "<<k<<" diff "<<my_diff<<" RMS "<<theRMS<<std::endl;
	  
	  counter++; 
	  myIndex = first_strip_index+counter-1;
	  if (counter>size[i]*LAYERS_con) counter=0;
	  myConnectFile <<"  "<<myIndex-1<<"  "<<my_diff<<"  "<<theRMS<<std::endl;
	  //flags for RMS and baseline
	  if (my_diff > 100.0)      flagConnect = 1; // ok
	  if (my_diff > 0.0 && my_diff<100.0) flagConnect = 2; //intermidiate state...
	  if (my_diff == 0.0 )      flagConnect = 3; //strip dead
	  if (my_diff > 1500.0)     flagConnect = 4; //strip noisy
	  
	  theRMS       = sqrt(fabs(my_diffSquare - my_diff*my_diff));
	  calib_evt.strip=k;
	  calib_evt.layer=j;
	  calib_evt.cham=i;
	  calib_evt.ddu=iii;
	  calib_evt.adcMin = adcMean_min[iii][i][j][k];
	  calib_evt.adcMax = adcMean_max[iii][i][j][k];
	  calib_evt.diff=my_diff;
	  calib_evt.RMS=theRMS;
	  calib_evt.flagConnect = flagConnect;
	  calibtree.Fill();
	  
	  //send constants to DB
	  /*
	    cn->obj[layer_id][k].resize(2);
	    cn->obj[layer_id][k][0] = my_diff;
	    cn->obj[layer_id][k][1] = theRMS;
	  */
	}
      }
    }
  }
  calibfile.Write();    
  calibfile.Close();  
}

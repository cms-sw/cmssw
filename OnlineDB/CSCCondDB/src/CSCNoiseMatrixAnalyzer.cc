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
#include "OnlineDB/CSCCondDB/interface/CSCNoiseMatrixAnalyzer.h"

CSCNoiseMatrixAnalyzer::CSCNoiseMatrixAnalyzer(edm::ParameterSet const& conf) {
  debug = conf.getUntrackedParameter<bool>("debug",false);
  eventNumber=0,evt=0,NChambers=0,Nddu=0,counterzero=0;
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
         
      for (unsigned int iDDU=0; iDDU<dduData.size(); ++iDDU) { 
	
	///get a reference to chamber data
	const std::vector<CSCEventData> & cscData = dduData[iDDU].cscData();
	//exclude empty events with no DMB/CFEB data
	//	if(dduData[iDDU].cscData().size()==0) continue;
	//	if(dduData[iDDU].cscData().size() !=0) evt++;   
	
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
	edm::LogInfo ("CSCNoiseMatrixAnalyzer")  << "end of event number " << eventNumber<<" and non-zero event "<<evt;
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
    
    if (lines==2){
      name=PSet;  
      std::cout<<name<<std::endl;
    }
  }
  std::string::size_type runNameStart = name.find("\"",0);
  std::string::size_type runNameEnd   = name.find("raw",0);
  std::string::size_type rootStart    = name.find("SCAPed",0);
  int nameSize = runNameEnd+2-runNameStart;
  int myRootSize = rootStart-runNameStart+11;
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
  std::ofstream myfile(myFileName,std::ios::out);
    
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

	  //Use averages of matrix elements per chamber in case of HUGE values from calibratin run
	  //ME+1/1 crates
	  if (new_crateID==2 && new_dmbID<4 && tmp[0]>20.0)  tmp[0] =7.86675;
	  if (new_crateID==2 && new_dmbID<4 && tmp[1]>20.0)  tmp[1] =2.07075;
	  if (new_crateID==2 && new_dmbID<4 && tmp[2]>20.0)  tmp[2] =6.93875;
	  if (new_crateID==2 && new_dmbID<4 && tmp[3]>20.0)  tmp[3] =1.42525;
	  if (new_crateID==2 && new_dmbID<4 && tmp[4]>20.0)  tmp[4] =2.51025;
	  if (new_crateID==2 && new_dmbID<4 && tmp[5]>20.0)  tmp[5] =7.93975;
	  if (new_crateID==2 && new_dmbID<4 && tmp[6]>20.0)  tmp[6] =0.94725;
	  if (new_crateID==2 && new_dmbID<4 && tmp[7]>20.0)  tmp[7] =2.39275;
	  if (new_crateID==2 && new_dmbID<4 && tmp[8]>20.0)  tmp[8] =6.46475;
	  if (new_crateID==2 && new_dmbID<4 && tmp[9]>20.0)  tmp[9] =1.86325;
	  if (new_crateID==2 && new_dmbID<4 && tmp[10]>20.0) tmp[10]=2.08025;
	  if (new_crateID==2 && new_dmbID<4 && tmp[11]>20.0) tmp[11]=6.67975;

	  //ME+1/2
	  if (new_crateID==2 && new_dmbID>3 && new_dmbID<8 && tmp[0]>20.0)  tmp[0] =9.118;
	  if (new_crateID==2 && new_dmbID>3 && new_dmbID<8 && tmp[1]>20.0)  tmp[1] =3.884;
	  if (new_crateID==2 && new_dmbID>3 && new_dmbID<8 && tmp[2]>20.0)  tmp[2] =7.771;
	  if (new_crateID==2 && new_dmbID>3 && new_dmbID<8 && tmp[3]>20.0)  tmp[3] =1.8225;
	  if (new_crateID==2 && new_dmbID>3 && new_dmbID<8 && tmp[4]>20.0)  tmp[4] =3.7505;
	  if (new_crateID==2 && new_dmbID>3 && new_dmbID<8 && tmp[5]>20.0)  tmp[5] =8.597;
	  if (new_crateID==2 && new_dmbID>3 && new_dmbID<8 && tmp[6]>20.0)  tmp[6] =1.651;
	  if (new_crateID==2 && new_dmbID>3 && new_dmbID<8 && tmp[7]>20.0)  tmp[7] =2.5225;
	  if (new_crateID==2 && new_dmbID>3 && new_dmbID<8 && tmp[8]>20.0)  tmp[8] =6.583;
	  if (new_crateID==2 && new_dmbID>3 && new_dmbID<8 && tmp[9]>20.0)  tmp[9] =1.5055;
	  if (new_crateID==2 && new_dmbID>3 && new_dmbID<8 && tmp[10]>20.0) tmp[10]=2.733;
	  if (new_crateID==2 && new_dmbID>3 && new_dmbID<8 && tmp[11]>20.0) tmp[11]=6.988;

	  //ME+1/3
	  if (new_crateID==2 && new_dmbID>7 && tmp[0]>20.0)  tmp[0] =9.5245;
	  if (new_crateID==2 && new_dmbID>7 && tmp[1]>20.0)  tmp[1] =3.2415;
	  if (new_crateID==2 && new_dmbID>7 && tmp[2]>20.0)  tmp[2] =7.6265;
	  if (new_crateID==2 && new_dmbID>7 && tmp[3]>20.0)  tmp[3] =1.7225;
	  if (new_crateID==2 && new_dmbID>7 && tmp[4]>20.0)  tmp[4] =3.6075;
	  if (new_crateID==2 && new_dmbID>7 && tmp[5]>20.0)  tmp[5] =8.7275;
	  if (new_crateID==2 && new_dmbID>7 && tmp[6]>20.0)  tmp[6] =1.663;
	  if (new_crateID==2 && new_dmbID>7 && tmp[7]>20.0)  tmp[7] =2.592;
	  if (new_crateID==2 && new_dmbID>7 && tmp[8]>20.0)  tmp[8] =7.5685;
	  if (new_crateID==2 && new_dmbID>7 && tmp[9]>20.0)  tmp[9] =1.7905;
	  if (new_crateID==2 && new_dmbID>7 && tmp[10]>20.0) tmp[10]=2.409;
	  if (new_crateID==2 && new_dmbID>7 && tmp[11]>20.0) tmp[11]=7.1495;

	  //ME+2/1
	  if (new_crateID==1 &&  new_dmbID<4 && !(tmp[0]<12.0 && tmp[0]>-10.0))   tmp[0] =9.06825;
	  if (new_crateID==1 &&  new_dmbID<4 && !(tmp[1]<7.0 && tmp[1]>-10.0))    tmp[1] =3.32025;
	  if (new_crateID==1 &&  new_dmbID<4 && !(tmp[2]<12.0 && tmp[2]>-10.0))   tmp[2] =7.52925;
	  if (new_crateID==1 &&  new_dmbID<4 && !(tmp[3]<8.0 && tmp[3]>-10.0))    tmp[3] =3.66125;
	  if (new_crateID==1 &&  new_dmbID<4 && !(tmp[4]<8.0 && tmp[4]>-10.0))    tmp[4] =3.39125;
	  if (new_crateID==1 &&  new_dmbID<4 && !(tmp[5]<14.0 && tmp[5]>-10.0))   tmp[5] =9.97625;
	  if (new_crateID==1 &&  new_dmbID<4 && !(tmp[6]<5.0 && tmp[6]>-10.0))    tmp[6] =1.32725;
	  if (new_crateID==1 &&  new_dmbID<4 && !(tmp[7]<7.0 && tmp[7]>-10.0))    tmp[7] =3.99025;
	  if (new_crateID==1 &&  new_dmbID<4 && !(tmp[8]<12.0 && tmp[8]>-10.0))   tmp[8] =8.10125;
	  if (new_crateID==1 &&  new_dmbID<4 && !(tmp[9]<6.0 && tmp[9]>-10.0))    tmp[9] =2.56456;
	  if (new_crateID==1 &&  new_dmbID<4 && !!(tmp[10]<7.0 && tmp[10]>-10.0))   tmp[10]=2.96625;
	  if (new_crateID==1 &&  new_dmbID<4 && (tmp[11]<11.0 && tmp[11]>-10.0))  tmp[11]=7.30925;

	  //ME+2/2
	  if (new_crateID==1 &&  new_dmbID>3 && !(tmp[0]<21.0 && tmp[0]>-10.0))   tmp[0] =16.7442;
	  if (new_crateID==1 &&  new_dmbID>3 && !(tmp[1]<12.0 && tmp[1]>-10.0))   tmp[1] =7.96925;
	  if (new_crateID==1 &&  new_dmbID>3 && !(tmp[2]<18.0 && tmp[2]>-10.0))   tmp[2] =14.1643;
	  if (new_crateID==1 &&  new_dmbID>3 && !(tmp[3]<9.0 && tmp[3]>-10.0))   tmp[3] =4.67975;
	  if (new_crateID==1 &&  new_dmbID>3 && !(tmp[4]<12.0 && tmp[4]>-10.0))   tmp[4] =8.44075;
	  if (new_crateID==1 &&  new_dmbID>3 && !(tmp[5]<21.0 && tmp[5]>-10.0))   tmp[5] =17.2243;
	  if (new_crateID==1 &&  new_dmbID>3 && !(tmp[6]<8.0 && tmp[6]>-10.0))    tmp[6] =3.68575;
	  if (new_crateID==1 &&  new_dmbID>3 && !(tmp[7]<12.0 && tmp[7]>-10.0))   tmp[7] =7.48825;
	  if (new_crateID==1 &&  new_dmbID>3 && !(tmp[8]<19.0 && tmp[8]>-10.0))   tmp[8] =14.4902;
	  if (new_crateID==1 &&  new_dmbID>3 && !(tmp[9]<9.0 && tmp[9]>-10.0))    tmp[9] =4.4482;
	  if (new_crateID==1 &&  new_dmbID>3 && !(tmp[10]<11.0 && tmp[10]>-10.0))  tmp[10]=6.47875;
	  if (new_crateID==1 &&  new_dmbID>3 && !(tmp[11]<19.0 && tmp[11]>-10.0))  tmp[11]=14.6733;	  

	  //ME+3/1
	  if (new_crateID==0 &&  new_dmbID<4 && tmp[0]>13.0)   tmp[0] =9.3495;
	  if (new_crateID==0 &&  new_dmbID<4 && tmp[1]>8.0)    tmp[1] =3.529;
	  if (new_crateID==0 &&  new_dmbID<4 && tmp[2]>13.0)   tmp[2] =7.8715;
	  if (new_crateID==0 &&  new_dmbID<4 && tmp[3]>8.0)    tmp[3] =3.8155;
	  if (new_crateID==0 &&  new_dmbID<4 && tmp[4]>8.0)    tmp[4] =3.858;
	  if (new_crateID==0 &&  new_dmbID<4 && tmp[5]>15.0)   tmp[5] =10.8205;
	  if (new_crateID==0 &&  new_dmbID<4 && tmp[6]>6.0)    tmp[6] =1.8585;
	  if (new_crateID==0 &&  new_dmbID<4 && tmp[7]>8.0)    tmp[7] =4.445;
	  if (new_crateID==0 &&  new_dmbID<4 && tmp[8]>12.0)   tmp[8] =8.0175;
	  if (new_crateID==0 &&  new_dmbID<4 && tmp[9]>7.0)    tmp[9] =3.29479;
	  if (new_crateID==0 &&  new_dmbID<4 && tmp[10]>8.0)   tmp[10]=3.625;
	  if (new_crateID==0 &&  new_dmbID<4 && tmp[11]>12.0)  tmp[11]=8.3895;

	  //ME+3/2
	  if (new_crateID==0 &&  new_dmbID>3 && tmp[0]>21.0)   tmp[0] =13.6193;
	  if (new_crateID==0 &&  new_dmbID>3 && tmp[1]>12.0)   tmp[1] =5.91025;
	  if (new_crateID==0 &&  new_dmbID>3 && tmp[2]>18.0)   tmp[2] =11.3842;
	  if (new_crateID==0 &&  new_dmbID>3 && tmp[3]>9.0)    tmp[3] =3.31775;
	  if (new_crateID==0 &&  new_dmbID>3 && tmp[4]>12.0)   tmp[4] =5.69775;
	  if (new_crateID==0 &&  new_dmbID>3 && tmp[5]>21.0)   tmp[5] =11.6652;
	  if (new_crateID==0 &&  new_dmbID>3 && tmp[6]>8.0)    tmp[6] =2.46175;
	  if (new_crateID==0 &&  new_dmbID>3 && tmp[7]>12.0)   tmp[7] =4.48325;
	  if (new_crateID==0 &&  new_dmbID>3 && tmp[8]>19.0)   tmp[8] =9.95725;
	  if (new_crateID==0 &&  new_dmbID>3 && tmp[9]>9.0)    tmp[9] =2.10561;
	  if (new_crateID==0 &&  new_dmbID>3 && tmp[10]>11.0)  tmp[10]=4.04625;
	  if (new_crateID==0 &&  new_dmbID>3 && tmp[11]>19.0)  tmp[11]=9.51625;	  


	  if (tmp[max]>3.0 && tmp[max]<100.0) flagMatrix = 1; // ok
	  if (tmp[max]>50.0)                  flagMatrix = 2; // warning too high
	  if (tmp[max]<-15.0)                 flagMatrix = 3; // warning too low
	  /*
	  if (isnan(tmp[0]))                  tmp[0]   = 1000.0;
	  if (isnan(tmp[1]))                  tmp[1]   = 1000.0;
	  if (isnan(tmp[2]))                  tmp[2]   = 1000.0;
	  if (isnan(tmp[3]))                  tmp[3]   = 1000.0;
	  if (isnan(tmp[4]))                  tmp[4]   = 1000.0;
	  if (isnan(tmp[5]))                  tmp[5]   = 1000.0;
	  if (isnan(tmp[6]))                  tmp[6]   = 1000.0;
	  if (isnan(tmp[7]))                  tmp[7]   = 1000.0;
	  if (isnan(tmp[8]))                  tmp[8]   = 1000.0;
	  if (isnan(tmp[9]))                  tmp[9]   = 1000.0;
	  if (isnan(tmp[10]))                 tmp[10]  = 1000.0;
	  if (isnan(tmp[11]))                 tmp[11]  = 1000.0;
	  */

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
	  
	  //std::cout<<"Chamber "<<i<<" Layer "<<j<<" strip "<<k<<" Matrix elements "<<tmp[max]<<std::endl;
	  
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

	myfile<<layer_id<<"  "<<k<<"  "<<tmp[0]<<"  "<<tmp[1]<<"  "<<tmp[3]<<"  "<<tmp[2]<<"  "<<tmp[4]<<"  "<<tmp[6]<<"  "<<tmp[5]<<"  "<<tmp[7]<<"  "<<tmp[9]<<"  "<<tmp[8]<<"  "<<tmp[10]<<"  "<<tmp[11]<<std::endl;
      }
    }
  }
  //}//myDDU
     
  //send data to DB
  dbon->cdbon_last_record("noisematrix",&record);
  std::cout<<"record "<<record<<" for run file "<<myname<<" saved "<<myTime<<std::endl;
  if(debug) dbon->cdbon_write(cn,"noisematrix",12,3498,myTime);
  std::cout<<"record "<<record<<" for run file "<<myname<<" saved "<<myTime<<std::endl;
  
  calibfile.Write();
  calibfile.Close();
}

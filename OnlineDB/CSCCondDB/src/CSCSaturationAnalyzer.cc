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

CSCSaturationAnalyzer::CSCSaturationAnalyzer(edm::ParameterSet const& conf) {
  debug = conf.getUntrackedParameter<bool>("debug",false);
  eventNumber=0,evt=0,Nddu=0;
  strip=0,misMatch=0,NChambers=0;
  i_chamber=0,i_layer=0,reportedChambers=0;
  length=1,gainSlope=-999.0,gainIntercept=-999.0;
  aVar=0.0,bVar=0.0;

  gain_vs_charge  = TH2F("Saturation ","ADC_vs_charge", 100,0,600,100,0,4000);
  gain1_vs_charge = TH2F("Saturation1","ADC_vs_charge", 100,0,600,100,0,4000);
  gain2_vs_charge = TH2F("Saturation2","ADC_vs_charge", 100,0,600,100,0,4000);
  gain3_vs_charge = TH2F("Saturation3","ADC_vs_charge", 100,0,600,100,0,4000);
  gain4_vs_charge = TH2F("Saturation4","ADC_vs_charge", 100,0,600,100,0,4000);
  gain5_vs_charge = TH2F("Saturation5","ADC_vs_charge", 100,0,600,100,0,4000);
  gain6_vs_charge = TH2F("Saturation6","ADC_vs_charge", 100,0,600,100,0,4000);


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
  
  for (int i=0; i<480; i++){
    newGain[i]     =0.0;
    newIntercept[i]=0.0;
    newChi2[i]     =0.0;
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
		
		//On the 20th event save
		if (evt%20 == 0 && (strip-1)%16 == (evt-1)/NUMMODTEN_sat){
		  int ten = int((evt-1)/20)%NUMBERPLOTTED_sat ;
		  maxmodten[ten][i_chamber][i_layer-1][strip-1] = adcMean_max[iDDU][i_chamber][i_layer-1][strip-1];
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
  string::size_type runNameEnd   = name.find("bin",0);
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
  cscmap *map = new cscmap();
  condbon *dbon = new condbon();
  
  //root ntuple information
  TCalibSaturationEvt calib_evt;
  TFile calibfile(myNewName, "RECREATE");
  TTree calibtree("Calibration","Saturation");
  calibtree.Branch("EVENT", &calib_evt, "slope/F:intercept/F:chi2/F:strip/I:layer/I:cham/I:id/I");
  gain_vs_charge.Write();
  gain1_vs_charge.Write();
  gain2_vs_charge.Write();
  gain3_vs_charge.Write();
  gain4_vs_charge.Write();
  gain5_vs_charge.Write();
  gain6_vs_charge.Write();
  
  for (int dduiter=0;dduiter<Nddu;dduiter++){
    for(int chamberiter=0; chamberiter<NChambers; chamberiter++){
      for (int cham=0;cham<NChambers;cham++){
	if (cham !=chamberiter) continue;
	
	//get chamber ID from DB mapping        
	int new_crateID = crateID[cham];
	int new_dmbID   = dmbID[cham];
	std::cout<<" Crate: "<<new_crateID<<" and DMB:  "<<new_dmbID<<std::endl;
	map->crate_chamber(new_crateID,new_dmbID,&chamber_id,&chamber_num,&sector);
	std::cout<<"Data is for chamber:: "<< chamber_id<<" in sector:  "<<sector<<std::endl;
	
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
		float sumOfX    = 0.0;
		float sumOfY    = 0.0;
		float sumOfXY   = 0.0;
		float sumx2     = 0.0;
		float gainSlope = 0.0;
		float gainIntercept = 0.0;
		float chi2      = 0.0;
		float chi2_sat  = 0.0;
		
		float charge[NUMBERPLOTTED_sat]={22.4, 44.8, 67.2, 89.6, 112.0, 134.4, 156.8, 179.2, 201.6, 224.0, 246.4, 268.8, 291.2, 313.6, 336.0, 358.4, 380.8, 403.2, 425.6, 448.0, 470.4, 492.8, 515.2, 537.6, 560.0};
		

		for(int ii=0; ii<NUMBERPLOTTED_sat; ii++){//numbers    
		  sumOfX  += charge[ii];
		  sumOfY  += maxmodten[ii][cham][j][k];
		  sumOfXY += (charge[ii]*maxmodten[ii][cham][j][k]);
		  sumx2   += (charge[ii]*charge[ii]);
		  myCharge[ii] = 22.4 +(22.4*ii);
		  mySatADC[ii] = maxmodten[ii][cham][j][k];
		  gain_vs_charge.Fill(myCharge[ii],maxmodten[ii][cham][j][k]);
		  if(cham==0 && j==1 && k==10) {
		    gain1_vs_charge.Fill(myCharge[ii],maxmodten[ii][cham][j][k]);
		    std::cout <<" this are my pairs "<<myCharge[ii]<<"   "<<maxmodten[ii][cham][j][k]<<std::endl;
		  }
		  if(cham==0 && j==1 && k==20) gain2_vs_charge.Fill(myCharge[ii],maxmodten[ii][cham][j][k]);
		  if(cham==0 && j==1 && k==30) gain3_vs_charge.Fill(myCharge[ii],maxmodten[ii][cham][j][k]);
		  if(cham==0 && j==1 && k==40) gain4_vs_charge.Fill(myCharge[ii],maxmodten[ii][cham][j][k]);
		  if(cham==0 && j==1 && k==50) gain5_vs_charge.Fill(myCharge[ii],maxmodten[ii][cham][j][k]);
		  if(cham==0 && j==1 && k==60) gain6_vs_charge.Fill(myCharge[ii],maxmodten[ii][cham][j][k]);
		  
		  float (*charge_ptr)[NUMBERPLOTTED_sat] = &myCharge;
		  float (*adc_ptr)[NUMBERPLOTTED_sat]    = &mySatADC;

		  //Fit parameters for straight line
		  gainSlope     = ((NUMBERPLOTTED_sat*sumOfXY) - (sumOfX * sumOfY))/((NUMBERPLOTTED_sat*sumx2) - (sumOfX*sumOfX));//k
		  gainIntercept = ((sumOfY*sumx2)-(sumOfX*sumOfXY))/((NUMBERPLOTTED_sat*sumx2)-(sumOfX*sumOfX));//m
		  
		  for(int ii=0; ii<NUMBERPLOTTED_sat; ii++){
		    chi2  += (maxmodten[ii][cham][j][k]-(gainIntercept+(gainSlope*charge[ii])))*(maxmodten[ii][cham][j][k]-(gainIntercept+(gainSlope*charge[ii])))/(NUMBERPLOTTED_sat*NUMBERPLOTTED_sat);

		    chi2_sat +=gainSlope -(aVar*charge[ii]/(bVar+charge[ii]));
		  }
		  
		  calib_evt.slope     = gainSlope;
		  calib_evt.intercept = gainIntercept;
		  calib_evt.chi2      = chi2;
		  calib_evt.strip     = k;
		  calib_evt.layer     = j;
		  calib_evt.cham      = cham;
		  
		  calibtree.Fill();
		  
		  cn->obj[layer_id][k].resize(3);
		  cn->obj[layer_id][k][0] = gainSlope;
		  cn->obj[layer_id][k][1] = gainIntercept;
		  cn->obj[layer_id][k][2] = chi2;
		}//number_plotted
	      }//strip
	    }//j loop
	  }//stripiter loop
	}//layiter loop
      }//cham 
    }//chamberiter
  }//dduiter

  //send data to DB
  dbon->cdbon_last_record("gains",&record);
  std::cout<<"Last gains record "<<record<<" for run file "<<myname<<" saved "<<myTime<<std::endl;
  if(debug) dbon->cdbon_write(cn,"gains",11,myTime);
  gain_vs_charge.Write();
  gain1_vs_charge.Write();
  gain2_vs_charge.Write();
  gain3_vs_charge.Write();
  gain4_vs_charge.Write();
  gain5_vs_charge.Write();
  gain6_vs_charge.Write();
  
  calibfile.Write();
  calibfile.Close();
}

/** 
 * Analyzer for reading CSC comapartor thresholds.
 * author O.Boeriu 17/11/06  
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
#include "DataFormats/CSCDigi/interface/CSCComparatorDigi.h"
#include "DataFormats/CSCDigi/interface/CSCComparatorDigiCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "IORawData/CSCCommissioning/src/FileReaderDDU.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDDUEventData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDCCEventData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCEventData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDMBHeader.h"
#include "OnlineDB/CSCCondDB/interface/CSCCompThreshAnalyzer.h"

CSCCompThreshAnalyzer::CSCCompThreshAnalyzer(edm::ParameterSet const& conf) {
  debug = conf.getUntrackedParameter<bool>("debug",false);
  eventNumber = 0,compadc=0;
  evt = 0,Nddu=0,misMatch=0,event=0;
  i_chamber=0,i_layer=0,reportedChambers =0;
  length = 1, NChambers=0;
  meanThresh=0.0;
  timebin=-999,mycompstrip=-999,comparator=0,compstrip=0;

  adc_vs_charge  = TH2F("CFEB Comparator"   ,"ADC_vs_charge", 100,0,300,100,0,2);

   for (int i=0; i<NUMBERPLOTTED_ct; i++){
    for (int j=0; j<CHAMBERS_ct; j++){
      for (int k=0; k<LAYERS_ct; k++){
	for (int l=0; l<STRIPS_ct; l++){
	  meanmod[i][j][k][l] = 0.0;
	}
      }
    }
  }

  for(int i=0;i<CHAMBERS_ct;i++){
    for(int j=0; j<LAYERS_ct; j++){
      for(int k=0; k<STRIPS_ct; k++){
	theMeanThresh[i][j][k] = 0.;
	arrayMeanThresh[i][j][k] = 0.;
	mean[i][j][k]=0.;
	meanTot[i][j][k]=0.;
      }
    }
  }


  for (int i=0; i<CHAMBERS_ct; i++){
    size[i]  = 0;
  }

}

void CSCCompThreshAnalyzer::analyze(edm::Event const& e, edm::EventSetup const& iSetup) {
  
  // These declarations create handles to the types of records that you want
  // to retrieve from event "e".
  //
   edm::Handle<CSCStripDigiCollection> strips;
   edm::Handle<CSCComparatorDigiCollection> comparators;
   
  // Pass the handle to the method "getByType", which is used to retrieve
  // one and only one instance of the type in question out of event "e". If
  // zero or more than one instance exists in the event an exception is thrown.
  //

   e.getByLabel("cscunpacker","MuonCSCStripDigi",strips);
   e.getByLabel("cscunpacker","MuonCSCComparatorDigi",comparators);

   for (CSCComparatorDigiCollection::DigiRangeIterator j=comparators->begin(); j!=comparators->end(); j++) {
     std::vector<CSCComparatorDigi>::const_iterator digiItr = (*j).second.first;
     std::vector<CSCComparatorDigi>::const_iterator last = (*j).second.second;
     for( ; digiItr != last; ++digiItr) {
       //digiItr->print();
       std::cout<<"This is comp "<<digiItr->getStrip()<<std::endl;
     }
   }


   edm::Handle<FEDRawDataCollection> rawdata;
   e.getByType(rawdata);
   event =e.id().event();
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
	 Nddu=dduData.size();
	 reportedChambers += dduData[iDDU].header().ncsc();
	 NChambers = cscData.size();
	 int repChambers = dduData[iDDU].header().ncsc();
	 std::cout << " Reported Chambers = " << repChambers <<"   "<<NChambers<< std::endl;
	 if (NChambers!=repChambers) {std::cout<< "misMatched size!!!" << std::endl; misMatch++;continue;}
	 	 
	 for (i_chamber=0; i_chamber<NChambers; i_chamber++) {//loop over all DMBs  
	   if (cscData[i_chamber].nclct()) {
	     CSCCLCTData & clctData = cscData[i_chamber].clctData();
	   }else {
	     std::cout<<" No CLCT!" <<std::endl;
	     continue;
	   }
	   CSCCLCTData & clctData = cscData[i_chamber].clctData();
	   for(i_layer = 1; i_layer <= 6; ++i_layer) {//loop over all layers in chambers
	     std::vector<CSCComparatorDigi> comp = clctData.comparatorDigis(i_layer);
	     
	     const CSCDMBHeader &thisDMBheader = cscData[i_chamber].dmbHeader();
	     
	     if (thisDMBheader.cfebAvailable()){//check that CFEB data exists
	       
	       dmbID[i_chamber]   = cscData[i_chamber].dmbHeader().dmbID(); //get DMB ID
	       crateID[i_chamber] = cscData[i_chamber].dmbHeader().crateID(); //get crate ID
	       if(crateID[i_chamber] == 255) continue; //255 is reserved for old crate
	      
	       for (unsigned int i=0; i<comp.size(); i++){//loop over CFEB comparator digis
		 size[i_chamber] = comp.size();
		 comparator      = comp[i].getComparator();
		 timebin         = comp[i].getTimeBin() ;
		 compstrip       = comp[i].getStrip();

		 //was for di-strip logic
		 //int this_comparator[4] = {4, 5, 6, 7};
		 // for (int iii=0; iii<40; iii++){
		 // 		   if ((compstrip == iii) && (comparator == this_comparator[0] || comparator == this_comparator[1])) {
		 // 		     mycompstrip = 0 + iii*2;
		 // 		   } else if ((compstrip == iii) && (comparator == this_comparator[2] || comparator == this_comparator[3])) {
		 // 		     mycompstrip = 1 + iii*2;
		 // 		   }
		 // 		 }

		 //is now for left=0,for right=1 halfstrip 
		 for (int iii=0; iii<40; iii++){
		   if ((compstrip == iii) && (comparator == 0)) {
		     mycompstrip = 0 + iii*2;
		   } else if ((compstrip == iii) && (comparator == 1)) {
		     mycompstrip = 1 + iii*2;
		   }
		 }
		 
		 mean[i_chamber][i_layer-1][mycompstrip] = comparator/5.;
		 std::cout<<" mean "<<mean[i_chamber][i_layer-1][mycompstrip]<<" compstrip "<<mycompstrip<<std::endl;
		 
	       }//end comp loop
	       
	       meanTot[i_chamber][i_layer-1][mycompstrip] +=mean[i_chamber][i_layer-1][mycompstrip]/25.;
	       
	       // On the 25th event and per CFEB
	       if (evt%25 == 0 && (mycompstrip)%16 == (evt-1)/NUMMOD_ct){
		 int tmp = int((evt-1)/25)% NUMBERPLOTTED_ct ;
		 std::cout<<" THIS IS tmp "<<tmp<<std::endl;
		 meanmod[tmp][i_chamber][i_layer-1][mycompstrip] = meanTot[i_chamber][i_layer-1][mycompstrip];

		 std::cout<<" meanval 25th event "<<meanmod[tmp][i_chamber][i_layer-1][mycompstrip]<<std::endl;
	       }
	     }//end if cfeb.available loop
	   }//end layer loop
	 }//end chamber loop

	 if((evt-1)%25==0){
	   //for(int iii=0;iii<DDU_sat;iii++){
	   for(int ii=0;ii<CHAMBERS_ct;ii++){
	     for(int jj=0;jj<LAYERS_ct;jj++){
	       for(int kk=0;kk<STRIPS_ct;kk++){
		 mean[ii][jj][kk]=0.0;
		 meanTot[ii][jj][kk]=0.0;
	       }
	     }
	   }
	 }
	 //}

	 eventNumber++;
	 edm::LogInfo ("CSCCompThreshAnalyzer")  << "end of event number " << eventNumber;
	 
       }//end DDU loop
     }
   }
}


CSCCompThreshAnalyzer::~CSCCompThreshAnalyzer(){
  //get time of Run file for DB transfer
  filein.open("../test/CSCcomp.cfg");
  filein.ignore(1000,'\n');
  
  while(filein != NULL){
    lines++;
    getline(filein,PSet);
      
    if (lines==3){
      name=PSet;  
      std::cout<<name<<std::endl;
    }
  }
  std::string::size_type runNameStart = name.find("\"",0);
  std::string::size_type runNameEnd   = name.find("raw",0);
  std::string::size_type rootStart    = name.find("CFEBComparator",0);
  int nameSize = runNameEnd+2-runNameStart;
  int myRootSize = rootStart-runNameStart+13;
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
  //condbon *dbon = new condbon();
 
 //root ntuple information
  TCalibComparatorEvt calib_evt;
  TFile calibfile(myNewName, "RECREATE");
  TTree calibtree("Calibration","CFEB Comparator");
  calibtree.Branch("EVENT", &calib_evt, "strip/I:layer/I:cham/I:id/I");

 
  for (int dduiter=0;dduiter<Nddu;dduiter++){
    for(int chamberiter=0; chamberiter<NChambers; chamberiter++){
      for (int cham=0;cham<NChambers;cham++){
	if (cham !=chamberiter) continue;

	//get chamber ID from DB mapping        
	int new_crateID = crateID[cham];
	int new_dmbID   = dmbID[cham];
	std::cout<<" Crate: "<<new_crateID<<" and DMB:  "<<new_dmbID<<std::endl;
	map->crate_chamber(new_crateID,new_dmbID,&chamber_id,&chamber_num,&sector,&first_strip_index,&strips_per_layer,&chamber_index);
	std::cout<<"Data is for chamber:: "<< chamber_id<<" in sector:  "<<sector<<std::endl;
	
	calib_evt.id=chamber_num;
	
	for (int layeriter=0; layeriter<LAYERS_ct; layeriter++){
	  for (int stripiter=0; stripiter<STRIPS_ct; stripiter++){

	    for (int j=0; j<LAYERS_ct; j++){//layer
	      if (j != layeriter) continue;

	      int layer_id=chamber_num+j+1;
	      if(sector==-100)continue;
	      cn->obj[layer_id].resize(size[cham]);
	      
	      for (int k=0; k<size[cham]; k++){//strip
		if (k != stripiter) continue;
		
		for (int st=0;st<NUMBERPLOTTED_ct;st++){
		  myCharge[st]  =0.0;
		  myCompProb[st]=0.0;
		}
		
		for(int ii=0; ii<NUMBERPLOTTED_ct; ii++){//numbers   
		  //start at 13mV,35 steps of 3mV;
		  myCharge[ii] = 13 +(3*ii);
		  myCompProb[ii] = meanmod[ii][cham][j][k];
		  adc_vs_charge.Fill(myCharge[ii],meanmod[ii][cham][j][k]);
		  meanThresh= meanmod[ii][cham][j][k];
		  std::cout<<"Ch "<<cham<<" Layer "<<j<<" strip "<<k<<" comparator threshold "<<meanThresh<<std::endl;	 
		}//numberplotted

		calib_evt.strip = k;
		calib_evt.layer = j;
		calib_evt.cham  = cham;

		calibtree.Fill();
	      }//strip
	    }//j loop
	  }//stripiter
	}//layeriter
      }//cham
    }//chamberiter
  }//dduiter

  //send data to DB
  //dbon->cdbon_last_record("comparator",&record);
  //std::cout<<"Last comparator record "<<record<<" for run file "<<myname<<" saved "<<myTime<<std::endl;
  //if(debug) dbon->cdbon_write(cn,"comparator",11,myTime);

  //write histograms 
  adc_vs_charge.Write();
  calibfile.Write();
  calibfile.Close();
}

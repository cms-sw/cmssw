/*
 * \file DTDataIntegrityTest.cc
 * 
 * $Date: 2007/03/15 18:33:46 $
 * $Revision: 1.0 $
 * \author S. Bolognesi - CERN
 *
 */

#include <DQM/DTMonitorClient/src/DTDataIntegrityTest.h>

//Framework
#include <DQMServices/Core/interface/MonitorElementBaseT.h>
#include <DataFormats/FEDRawData/interface/FEDNumbering.h>
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <iostream>
#include <string>

using namespace std;
using namespace edm;

DTDataIntegrityTest::DTDataIntegrityTest(const ParameterSet& ps){
  
  debug = ps.getUntrackedParameter<bool>("debug", "false");
  if(debug)
    cout<<"[DTDataIntegrityTest]: Constructor"<<endl;

  parameters = ps;
  
  dbe = Service<DaqMonitorBEInterface>().operator->();
  dbe->setVerbose(1);
}

DTDataIntegrityTest::~DTDataIntegrityTest(){
  if(debug)
    cout << "DataIntegrityTest: analyzed " << nevents << " events" << endl;
}

void DTDataIntegrityTest::beginJob(const edm::EventSetup& context){
  if(debug)
    cout<<"[DTtTrigCalibrationTest]: BeginJob"<<endl;

  nevents = 0;
}


void DTDataIntegrityTest::endJob(){

  if(debug)
    cout<<"[DTDataIntegrityTest] endjob called!"<<endl;

  if ( parameters.getUntrackedParameter<bool>("writeHisto", true) ) 
    dbe->save(parameters.getUntrackedParameter<string>("outputFile", "DTDataIntegrityTest.root"));
  
  dbe->rmdir("DT/Tests/DTDataIntegrity");
}

void DTDataIntegrityTest::bookHistos(string histoType, int dduId){
  stringstream dduId_s; dduId_s << dduId;
  dbe->setCurrentFolder("DT/Test/FED" + dduId_s.str());
  string histoName;

  if(histoType == "TTSValues_Percent"){
    histoName = "FED" + dduId_s.str() + histoType;
    (dduHistos[histoType])[dduId] = dbe->book1D(histoName,histoName,7,0,7);
    ((dduHistos[histoType])[dduId])->setBinLabel(1,"disconnected",1);	
    ((dduHistos[histoType])[dduId])->setBinLabel(2,"warning overflow",1);	
    ((dduHistos[histoType])[dduId])->setBinLabel(3,"out of synch",1);	
    ((dduHistos[histoType])[dduId])->setBinLabel(4,"busy",1);	
    ((dduHistos[histoType])[dduId])->setBinLabel(5,"ready",1);	
    ((dduHistos[histoType])[dduId])->setBinLabel(6,"error",1);	
    ((dduHistos[histoType])[dduId])->setBinLabel(7,"disconnected",1);	
  }
}

void DTDataIntegrityTest::analyze(const Event& e, const EventSetup& context){ 
  nevents++;
  if (nevents%1 == 0 && debug) 
    cout<<"[DTDataIntegrityTest]: "<<nevents<<" updates"<<endl;
  
  //Loop on FED id
  for (int dduId=FEDNumbering::getDTFEDIds().first; dduId<=FEDNumbering::getDTFEDIds().second; ++dduId){
    if(debug)
      cout<<"[DTDataIntegrityTest]:FED Id: "<<dduId<<endl;

    string histoType;
    //1D histo: % of tts values 
    MonitorElement * tts_histo = dbe->get(getMEName("DDUTTSValues",dduId));
    if (tts_histo) {
        if(debug)
	  cout<<"[DTDataIntegrityTest]:histo DDUTTSValues found"<<endl;

	histoType = "TTSValues_Percent";   
	if (dduHistos[histoType].find(dduId) == dduHistos[histoType].end()) {
	  bookHistos(histoType,dduId);
	} 
	for(int i=1;i<8;i++){
	  (dduHistos.find(histoType)->second).find(dduId)->second->
	    setBinContent(i,tts_histo->getBinContent(i)/tts_histo->getEntries());
	}

	//Check if there are too many events with wrong tts value
	double alert_tts1 = 0.5, alert_tts4 = 0.5, alert_tts20 = 0.5;
	if((dduHistos.find(histoType)->second).find(dduId)->second->getBinContent(2) > alert_tts1)
	  cout<<"[DTDataIntegrityTest]:WARNING: "<<
	    (dduHistos.find(histoType)->second).find(dduId)->second->getBinContent(2)<<" events with tts value = 1"<<endl;

   	if(((dduHistos.find(histoType)->second).find(dduId)->second->getBinContent(1) +
	    (dduHistos.find(histoType)->second).find(dduId)->second->getBinContent(3)) > alert_tts20 )
	  cout<<"[DTDataIntegrityTest]:WARNING: "<<
	    ((dduHistos.find(histoType)->second).find(dduId)->second->getBinContent(1) +
	     (dduHistos.find(histoType)->second).find(dduId)->second->getBinContent(3))<<" events with tts value = 2 or 0"<<endl;

	if((dduHistos.find(histoType)->second).find(dduId)->second->getBinContent(5) > alert_tts4)
	  cout<<"[DTDataIntegrityTest]:WARNING: more then "<<
	    (dduHistos.find(histoType)->second).find(dduId)->second->getBinContent(5)<<" events with tts value = 4"<<endl;
	//FIXME: how to notify this warning in the graphic display?
         }

    //Check if the list of ROS is compatible with the channels enabled
    MonitorElement * ros_histo = dbe->get(getMEName("DDUChannelStatus",dduId));
    if (ros_histo) {
        if(debug)
	  cout<<"[DTDataIntegrityTest]:histo DDUChannelStatus found"<<endl;

 	for(int i=1;i<13;i++){
	  if(ros_histo->getBinContent(1,i) != ros_histo->getBinContent(9,i))
	    cout<<"[DTDataIntegrityTest]:WARNING: ROS"<<i<<" in "<<tts_histo->getBinContent(9,i)<<" events"<<endl
		<<"               but channel"<<i<<" enabled in "<<tts_histo->getBinContent(1,i)<<" events"<<endl;
	  //FIXME: how to notify this warning in the graphic display?
	}
    }
  }
  //Save MEs in a root file
  if ((nevents%parameters.getUntrackedParameter<int>("saveResultsFrequency", 5)==0) && 
      (parameters.getUntrackedParameter<bool>("writeHisto", true)) ) 
    dbe->save(parameters.getUntrackedParameter<string>("outputFile", "DataIntegrityTest.root"));  
}

string DTDataIntegrityTest::getMEName(string histoType, int FEDId){
  //Use the DDU name to find the ME
  stringstream dduID_s; dduID_s << FEDId;
  string folderName ="Collector/FU0/DT/FED" + dduID_s.str(); 

  string histoName = folderName + "/FED" + dduID_s.str() + "_" + histoType;
  return histoName;
}

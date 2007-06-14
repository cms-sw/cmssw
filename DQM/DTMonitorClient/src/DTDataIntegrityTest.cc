/*
 * \file DTDataIntegrityTest.cc
 * 
 * $Date: 2007/05/23 12:36:55 $
 * $Revision: 1.7 $
 * \author S. Bolognesi - CERN
 *
 */

#include <DQM/DTMonitorClient/src/DTDataIntegrityTest.h>

//Framework
#include <DQMServices/Core/interface/MonitorElementBaseT.h>
#include <DataFormats/FEDRawData/interface/FEDNumbering.h>
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <string>

#include "TNamed.h"

using namespace std;
using namespace edm;

DTDataIntegrityTest::DTDataIntegrityTest(const ParameterSet& ps){
  
  edm::LogVerbatim ("dataIntegrity") << "[DTDataIntegrityTest]: Constructor";

  nTimeBin =  ps.getUntrackedParameter<int>("nTimeBin", 10);
  doTimeHisto =  ps.getUntrackedParameter<bool>("doTimeHisto", true);

  parameters = ps;

  dbe = Service<DaqMonitorBEInterface>().operator->();
  dbe->setVerbose(1);
}

DTDataIntegrityTest::~DTDataIntegrityTest(){

  edm::LogVerbatim ("dataIntegrity") << "DataIntegrityTest: analyzed " << nevents << " events";

}

void DTDataIntegrityTest::beginJob(const edm::EventSetup& context){

  edm::LogVerbatim ("dataIntegrity") << "[DTtTrigCalibrationTest]: BeginJob";

  nevents = 0;
}


void DTDataIntegrityTest::endJob(){

  edm::LogVerbatim ("dataIntegrity") <<"[DTDataIntegrityTest] endjob called!";

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

void DTDataIntegrityTest::bookTimeHistos(string histoType, int dduId, int evNumber){
  stringstream dduId_s; dduId_s << dduId;
  stringstream evNumber_s; evNumber_s << evNumber;
  string histoName;
  edm::LogVerbatim ("dataIntegrity") <<"Booking time histo "<<histoType<<" for ddu "<<dduId<<" from event "<<evNumber;

  //Counter for x bin in the timing histos
  counter = 1;//assuming synchronized booking for all histo VS time

  if(histoType == "TTSVSTime"){
    dbe->setCurrentFolder("DT/Test/FED" + dduId_s.str()+ "/TimeInfo/TTSVSTime");
    histoName = "FED" + dduId_s.str() + "_" + histoType + "_disconn_Ev" + evNumber_s.str();
    ((dduVectorHistos[histoType])[dduId]).push_back(dbe->book1D(histoName,histoName,nTimeBin,evNumber,evNumber+nTimeBin));
    histoName = "FED" + dduId_s.str() + histoType + "_overflow_Ev" + evNumber_s.str();
    ((dduVectorHistos[histoType])[dduId]).push_back(dbe->book1D(histoName,histoName,nTimeBin,evNumber,evNumber+nTimeBin));
    histoName = "FED" + dduId_s.str() + histoType + "_outSynch_Ev" + evNumber_s.str();
    ((dduVectorHistos[histoType])[dduId]).push_back(dbe->book1D(histoName,histoName,nTimeBin,evNumber,evNumber+nTimeBin));
    histoName = "FED" + dduId_s.str() + histoType + "_busy_Ev" + evNumber_s.str();
    ((dduVectorHistos[histoType])[dduId]).push_back(dbe->book1D(histoName,histoName,nTimeBin,evNumber,evNumber+nTimeBin));
    histoName = "FED" + dduId_s.str() + histoType + "_ready_Ev" + evNumber_s.str();
    ((dduVectorHistos[histoType])[dduId]).push_back(dbe->book1D(histoName,histoName,nTimeBin,evNumber,evNumber+nTimeBin));
    histoName = "FED" + dduId_s.str() + histoType + "_error_Ev" + evNumber_s.str();
    ((dduVectorHistos[histoType])[dduId]).push_back(dbe->book1D(histoName,histoName,nTimeBin,evNumber,evNumber+nTimeBin));
    histoName = "FED" + dduId_s.str() + histoType + "_disconnect_Ev" + evNumber_s.str();
    ((dduVectorHistos[histoType])[dduId]).push_back(dbe->book1D(histoName,histoName,nTimeBin,evNumber,evNumber+nTimeBin));
  }
  else if(histoType == "ROSVSTime"){
    dbe->setCurrentFolder("DT/Test/FED" + dduId_s.str()+ "/TimeInfo/ROSVSTime");
    histoName = "FED" + dduId_s.str() + "_" + histoType + "_Ev" + evNumber_s.str();
    (dduHistos[histoType])[dduId] = dbe->book1D(histoName,histoName,nTimeBin,evNumber,evNumber+nTimeBin);
  }
  else if(histoType == "EventLenghtVSTime"){
    dbe->setCurrentFolder("DT/Test/FED" + dduId_s.str()+ "/TimeInfo/EvLenghtVSTime");
    histoName = "FED" + dduId_s.str() + "_" + histoType + "_Ev" +  evNumber_s.str();
    (dduHistos[histoType])[dduId] = dbe->book1D(histoName,histoName,nTimeBin,evNumber,evNumber+nTimeBin);
  }
  else if(histoType == "FIFOVSTime"){
    dbe->setCurrentFolder("DT/Test/FED" + dduId_s.str()+ "/TimeInfo/FIFOVSTime");
    histoName = "FED" + dduId_s.str() + "_" + histoType + "_Input1_Ev" + evNumber_s.str();
    ((dduVectorHistos[histoType])[dduId]).push_back(dbe->book1D(histoName,histoName,nTimeBin,evNumber,evNumber+nTimeBin));
    histoName = "FED" + dduId_s.str() + "_" + histoType + "_Input2_Ev" + evNumber_s.str();
    ((dduVectorHistos[histoType])[dduId]).push_back(dbe->book1D(histoName,histoName,nTimeBin,evNumber,evNumber+nTimeBin));
    histoName = "FED" + dduId_s.str() + "_" + histoType + "_Input3_Ev" + evNumber_s.str();
    ((dduVectorHistos[histoType])[dduId]).push_back(dbe->book1D(histoName,histoName,nTimeBin,evNumber,evNumber+nTimeBin));
    histoName = "FED" + dduId_s.str() + "_" + histoType + "_L1A1_Ev" + evNumber_s.str();
    ((dduVectorHistos[histoType])[dduId]).push_back(dbe->book1D(histoName,histoName,nTimeBin,evNumber,evNumber+nTimeBin));
    histoName = "FED" + dduId_s.str() + "_" + histoType + "_L1A2_Ev" + evNumber_s.str();
    ((dduVectorHistos[histoType])[dduId]).push_back(dbe->book1D(histoName,histoName,nTimeBin,evNumber,evNumber+nTimeBin));
    histoName = "FED" + dduId_s.str() + "_" + histoType + "_L1A3_Ev" + evNumber_s.str();
    ((dduVectorHistos[histoType])[dduId]).push_back(dbe->book1D(histoName,histoName,nTimeBin,evNumber,evNumber+nTimeBin));
    histoName = "FED" + dduId_s.str() + "_" + histoType + "_Output_Ev" + evNumber_s.str();
    ((dduVectorHistos[histoType])[dduId]).push_back(dbe->book1D(histoName,histoName,nTimeBin,evNumber,evNumber+nTimeBin));
  }
} 


void DTDataIntegrityTest::analyze(const Event& e, const EventSetup& context){ 

  // counts number of updats (online mode) or number of events (standalone mode)
  nevents++;
  // if running in standalone perform diagnostic only after a reasonalbe amount of events
  if ( parameters.getUntrackedParameter<bool>("runningStandalone", false) && 
       nevents%parameters.getUntrackedParameter<int>("diagnosticPrescale", 1000) != 0 ) return;

  int evNumber = e.id().event();
  stringstream evNumber_s; evNumber_s << evNumber;
 
  edm::LogVerbatim ("dataIntegrity") <<"[DTDataIntegrityTest]: "<<nevents<<" updates";
  edm::LogVerbatim ("dataIntegrity") <<"[DTDataIntegrityTest]: "<<evNumber<<" event number";

  if(nevents%nTimeBin == 0){
    edm::LogVerbatim ("dataIntegrity") <<"[DTDataIntegrityTest]: saving all histos";
    dbe->save(parameters.getUntrackedParameter<string>("outputFile", "DTDataIntegrityTest.root"));
   }
  
  //Counter for x bin in the timing histos
  counter++;

  //Loop on FED id
  for (int dduId=FEDNumbering::getDTFEDIds().first; dduId<=FEDNumbering::getDTFEDIds().second; ++dduId){
    edm::LogVerbatim ("dataIntegrity") <<"[DTDataIntegrityTest]:FED Id: "<<dduId;
 
    //Each nTimeBin onUpdate remove timing histos and book a new bunch of them
    stringstream dduId_s; dduId_s << dduId;
    if(nevents%nTimeBin == 0 && doTimeHisto){
      edm::LogVerbatim ("dataIntegrity") <<"[DTDataIntegrityTest]: booking a new bunch of time histos";
      dbe->rmdir("DT/FED" + dduId_s.str() + "/TimeInfo");
      (dduVectorHistos.find("TTSVSTime")->second).find(dduId)->second.clear();
      bookTimeHistos("TTSVSTime",dduId, evNumber);
      bookTimeHistos("ROSVSTime",dduId, evNumber);
      bookTimeHistos("EventLenghtVSTime",dduId,evNumber);
      bookTimeHistos("FIFOVSTime",dduId,evNumber);
  }

    string histoType;
    //1D histo: % of tts values 
    MonitorElement * tts_histo = dbe->get(getMEName("TTSValues",dduId));
    if (tts_histo) {
        edm::LogVerbatim ("dataIntegrity") <<"[DTDataIntegrityTest]:histo DDUTTSValues found";

	histoType = "TTSValues_Percent";   
	if (dduHistos[histoType].find(dduId) == dduHistos[histoType].end()) {
	  bookHistos(histoType,dduId);
	} 
	//Loop on possible tts values
	for(int i=1;i<8;i++){
	  (dduHistos.find(histoType)->second).find(dduId)->second->
	    setBinContent(i,tts_histo->getBinContent(i)/tts_histo->getEntries());

	  if(doTimeHisto){
	    //Fill timing histos and set x label with event number
	    if( dduVectorHistos["TTSVSTime"].find(dduId) == dduVectorHistos["TTSVSTime"].end() ){
	      bookTimeHistos("TTSVSTime",dduId,evNumber); 
	    }
	    (dduVectorHistos.find("TTSVSTime")->second).find(dduId)->second[i-1]->
	      setBinContent(counter,tts_histo->getBinContent(i)/tts_histo->getEntries());
	    (dduVectorHistos.find("TTSVSTime")->second).find(dduId)->second[i-1]->
	      setBinLabel(counter, evNumber_s.str(), 1);
	  }
	}

	//Check if there are too many events with wrong tts value
	double alert_tts1 = 0.5, alert_tts4 = 0.5, alert_tts20 = 0.5;
	if((dduHistos.find(histoType)->second).find(dduId)->second->getBinContent(2) > alert_tts1)
	  edm::LogWarning ("dataIntegrity") <<"[DTDataIntegrityTest]:WARNING: "<<
	    (dduHistos.find(histoType)->second).find(dduId)->second->getBinContent(2)<<" % events with warning overflow";

   	if(((dduHistos.find(histoType)->second).find(dduId)->second->getBinContent(1) +
	    (dduHistos.find(histoType)->second).find(dduId)->second->getBinContent(3)) > alert_tts20 )
	  edm::LogWarning ("dataIntegrity") <<"[DTDataIntegrityTest]:WARNING: "<<
	    ((dduHistos.find(histoType)->second).find(dduId)->second->getBinContent(1) +
	     (dduHistos.find(histoType)->second).find(dduId)->second->getBinContent(3))<<" % events with out of synch or disconnected";

	if((dduHistos.find(histoType)->second).find(dduId)->second->getBinContent(4) > alert_tts4)
	  edm::LogWarning ("dataIntegrity") <<"[DTDataIntegrityTest]:WARNING: "<<
	    (dduHistos.find(histoType)->second).find(dduId)->second->getBinContent(4)<<" % events with busy";
	//FIXME: how to notify this warning in a LogFile?
         }

    //Check if the list of ROS is compatible with the channels enabled
    MonitorElement * ros_histo = dbe->get(getMEName("ROSStatus",dduId));
    if (ros_histo) {
        edm::LogVerbatim ("dataIntegrity") <<"[DTDataIntegrityTest]:histo DDUChannelStatus found";

 	for(int i=1;i<13;i++){
	  if(ros_histo->getBinContent(1,i) != ros_histo->getBinContent(9,i))
	    edm::LogError ("dataIntegrity") <<"[DTDataIntegrityTest]:WARNING: ROS"<<i<<" in "
					    <<tts_histo->getBinContent(9,i)<<" events"<<endl
					    <<"               but channel"<<i<<" enabled in "
					    <<tts_histo->getBinContent(1,i)<<" events";
	  //FIXME: how to notify this warning in a LogFile?
	}
    }
    //Monitor the number of ROS VS time
     MonitorElement * rosNumber_histo = dbe->get(getMEName("ROSList",dduId));
    if (rosNumber_histo && doTimeHisto) {
      edm::LogVerbatim ("dataIntegrity") <<"[DTDataIntegrityTest]:histo DDUROSList found";

      double rosNumber_mean = rosNumber_histo->getMean();
      //Fill timing histos and set x label with event number
      histoType = "ROSVSTime";
      if (dduHistos[histoType].find(dduId) == dduHistos[histoType].end()) {
	bookTimeHistos(histoType,dduId,evNumber);
      }
      (dduHistos.find(histoType)->second).find(dduId)->second->setBinContent(counter,rosNumber_mean);
      (dduHistos.find(histoType)->second).find(dduId)->second->setBinLabel(counter, evNumber_s.str(), 1);
   }

    //Monitor the event lenght VS time
     MonitorElement * evLenght_histo = dbe->get(getMEName("EventLenght",dduId));
     if (evLenght_histo && doTimeHisto) {
       edm::LogVerbatim ("dataIntegrity") <<"[DTDataIntegrityTest]:histo DDUEventLenght found";

      double evLenght_mean = evLenght_histo->getMean();
      //Fill timing histos and set x label with event number
      histoType = "EventLenghtVSTime";
      if (dduHistos[histoType].find(dduId) == dduHistos[histoType].end()) {
	bookTimeHistos(histoType,dduId,evNumber);
      }
      (dduHistos.find(histoType)->second).find(dduId)->second->setBinContent(counter,evLenght_mean);
      (dduHistos.find(histoType)->second).find(dduId)->second->setBinLabel(counter, evNumber_s.str(), 1);

      //FIXME: it does not reset the histo!!!
      MonitorElementT<TNamed>* ob = dynamic_cast<MonitorElementT<TNamed>*>( const_cast<MonitorElement*>(evLenght_histo) );
      if( ob ) {
  	ob->Reset();
      }
      
      
     }

     //Monitor the FIFO occupancy VS time 
     MonitorElement * fifo_histo = dbe->get(getMEName("FIFOStatus",dduId));
     if (fifo_histo && doTimeHisto) {
       edm::LogVerbatim ("dataIntegrity") <<"[DTDataIntegrityTest]:histo DDUFIFOStatus found";
       
       //Fill timing histos and set x label with event number
       histoType = "FIFOVSTime";
       if (dduVectorHistos[histoType].find(dduId) == dduVectorHistos[histoType].end()) {
	 bookTimeHistos(histoType,dduId,evNumber);
       }
       for(int i=1;i<8;i++){
	 (dduVectorHistos.find("FIFOVSTime")->second).find(dduId)->second[i-1]->
	   setBinContent(counter,(fifo_histo->getBinContent(i,1) + 2*(fifo_histo->getBinContent(i,2)))/fifo_histo->getEntries());
	 (dduVectorHistos.find("FIFOVSTime")->second).find(dduId)->second[i-1]->
	   setBinLabel(counter, evNumber_s.str(), 1);
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
  string folderName ="Collector/FU0/DT/DataIntegrity/FED" + dduID_s.str(); 

  string histoName = folderName + "/FED" + dduID_s.str() + "_" + histoType;
  return histoName;
}

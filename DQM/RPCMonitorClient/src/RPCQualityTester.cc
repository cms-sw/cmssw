/** \file
 *
 *  Implementation of RPCQualityTester
 *
 *  $Date: 2006/02/02 15:50:32 $
 *  $Revision: 1.2 $
 *  \author Ilaria Segoni
 */
#include "DQM/RPCMonitorClient/interface/RPCQualityTester.h"
#include "DQM/RPCMonitorClient/interface/RPCQualityTestTypes.h"
#include "DQMServices/QualityTests/interface/QCriterionRoot.h"

#include<iostream>
#include <string>

//
RPCQualityTester::RPCQualityTester()
{
  
  printout=true;
  logFile.open("CSCDQMClient.log");
  
  if(printout) logFile <<"An RPC Quality Tester Is being Created"<<std::endl;

}

//
RPCQualityTester::~RPCQualityTester(){ 
  logFile.close();
}



//
void RPCQualityTester::SetupTests(MonitorUserInterface * mui){
  if(printout) logFile<<"In RPCQualityTester::SetupTests(...)"<<std::endl;  
  this->SetupTestsFromTextFile(mui);
}


void RPCQualityTester::RunTests(MonitorUserInterface * mui){
  this->LinkTeststoME(mui);
  //this->AttachRunTests(mui);

}

//
void RPCQualityTester::SetupTestsFromTextFile(MonitorUserInterface * mui ){ 


  if(printout) logFile<<"In RPCQualityTester::SetupTestsFromTextFiles\n...\n"<<std::endl;
  FILE* TestsFile;
  char  fileName[128];
  sprintf(fileName,"QualityTests.db");
  TestsFile= fopen(fileName,"r");

  char  TestType[20];
  char  TestName[20];
  int   OnOff;
  float WarningLevel=0;
  float params[5];
  
  for(int ii=0; ii<5; ii++) params[ii]=0.0;

  int file_status=1;
 
  do{    
     file_status=fscanf(TestsFile,"%s%s%d%f%f%f%f%f%f\n",
                         &TestType,&TestName,&OnOff,&WarningLevel,
			 &params[0],&params[1],&params[2],&params[3],&params[4]);
			 
    if(file_status != EOF){			 
    
           if(printout)  logFile<<"\nReading Quality Tests Configuration: "
                          <<TestType<<" "<<TestName <<" "<<OnOff<<" "<<WarningLevel<<" "
	                  <<params[0]<<" "<<params[1]<<" "<<params[2]
	                  <<" "<<params[3]<<" "<<params[4]<< std::endl;
	   if(OnOff){
           	this->ConfigureTest(TestType,mui,TestName,WarningLevel,params);    
	   }
    }

  }while( file_status!=EOF ); 



}


void RPCQualityTester::LinkTeststoME(MonitorUserInterface * mui){ 


  if(printout) logFile<<"\nIn RPCQualityTester::LinkTeststoME"<<std::endl;
  FILE* TestsMEsFile;
  char  fileName[128];
  sprintf(fileName,"CSCTestsMEs.db");
  TestsMEsFile= fopen(fileName,"r");
  
  
  char TestName[20];
  char MEName[20];
  
  int file_status=1;
  
  do{    
      file_status=fscanf(TestsMEsFile,"%s%s\n",&TestName,&MEName);
			 
      if(file_status != EOF){			 
            if(printout) logFile<<"\nRequesting ME: "<<MEName<<"and attaching test: " <<TestName<<std::endl;
	    mui->subscribe(MEName);
            mui->useQTest(MEName, TestName);
	    
	    std::vector<std::string>  MEList;
	    if( qTestToMEMap.find(TestName) == qTestToMEMap.end()){
            if(printout) logFile<<"nuovo " <<TestName <<" "<<MEName << std::endl;
 	      MEList.push_back(MEName);
	      qTestToMEMap[TestName]=MEList;
	    }else{
            if(printout) logFile<<"trovato " <<TestName <<" "<<MEName << std::endl;
	      MEList=qTestToMEMap[TestName];
	      MEList.push_back(MEName);
             if(printout) logFile<<"ME's for test " <<TestName <<": "<<MEList.size() << std::endl;
	    }	    
      }	

  }while( file_status!=EOF ); 



}

void RPCQualityTester::SetContentsXRangeROOTTest(MonitorUserInterface * mui, char  TestName[20], float  WarningLevel, float  params[5]){

  if(printout) logFile<<"In RPCQualityTester::SetContentsXRangeROOTTest, configuring "<<TestName<<" test."<<std::endl;
	
  QCriterion * qc1 = mui->createQTest(ContentsXRangeROOT::getAlgoName(),TestName);
  MEContentsXRangeROOT * me_qc1 = (MEContentsXRangeROOT *) qc1;
  me_qc1->setAllowedXRange(params[0],params[1]);
  me_qc1->setWarningProb(WarningLevel);
 
}


void RPCQualityTester::SetContentsYRangeROOTTest(MonitorUserInterface * mui, char  TestName[20], float  WarningLevel, float  params[5]){

  if(printout) logFile<<"In RPCQualityTester::SetContentsYRangeROOTTest, configuring "<<TestName<<" test"<<std::endl;
	
  qTests.push_back(TestName);
  QCriterion * qc1 = mui->createQTest(ContentsYRangeROOT::getAlgoName(),TestName);
  MEContentsYRangeROOT * me_qc1 = (MEContentsYRangeROOT *) qc1;
  me_qc1->setAllowedYRange(params[0],params[1]);
  me_qc1->setWarningProb(WarningLevel);

}


void RPCQualityTester::AttachRunTests(MonitorUserInterface * mui){

if(printout) logFile<<"\nIn RPCQualityTester::AttachRunTests(...)"<<std::endl;

for (std::map<std::string, std::vector<std::string> >::iterator testsMap=qTestToMEMap.begin(); 
			   testsMap!=qTestToMEMap.end();++testsMap)	   
    {
     
       std::string testName=testsMap->first;
       std::vector<std::string> MElist=testsMap->second;
       if(printout) logFile<<"Number of ME's for Test "<< testName <<": "<<MElist.size()<<std::endl;

      	     for(std::vector<std::string>::iterator list = MElist.begin(); list != MElist.end(); ++list){
      	    	  std::string meName = *(list);
      	    	  if(printout) logFile<<"Attaching Test "<< testName <<" to ME "<<meName<<std::endl;
      	    	  mui->useQTest(meName, testName);
      	     }
    }
}



std::pair<std::string,std::string> RPCQualityTester::CheckTestsGlobal(MonitorUserInterface * mui) 
{
  if(printout) logFile << "\nChecking Global Status of Quiality Tests" << std::endl;
  std::pair<std::string,std::string> statement;
  
	int status = 0;
	
	status= mui->getSystemStatus();
        if(printout) logFile << "Possible states: successful "<< dqm::qstatus::STATUS_OK<<", error:  " 
	<< dqm::qstatus::ERROR<<
	", warning:  "<< dqm::qstatus::WARNING<<
	", Other: "<< dqm::qstatus::OTHER<<std::endl;
        if(printout) logFile << "The STATUS IS " << status<<std::endl;
	
	switch(status)
	  {
	  case dqm::qstatus::ERROR:
	    logFile << " Error(s)";
	    statement.first ="Errors detected in quality tests";
	    statement.second="red";
	    break;
	  case dqm::qstatus::WARNING:
	    logFile << " Warning(s)";
	    statement.first ="Warnings detected in quality tests";
	    statement.second="orange";
	    break;
	  case dqm::qstatus::OTHER:
	    logFile << " Some tests did not run";
	    statement.first="Some tests did not run";
	    statement.second="black";
	    break; 
	  default:
	    logFile << " No problems";
	    statement.first="No problems detected in quality tests ";
	    statement.second="green";
	  }
	    logFile << "\n";

  return statement;
}

 
void RPCQualityTester::ConfigureTest(char TestType[20], MonitorUserInterface * mui, char  TestName[20], float  WarningLevel, float  params[5]){
  
 qTests.push_back(TestName);
 
 logFile << "Setting up test "<<TestName<<std::endl;
 
 if(!std::strcmp(TestType,rpc_dqm::qTestType::XRangeContent.c_str())) this->SetContentsXRangeROOTTest(mui,TestName,WarningLevel,params);    
 if(!std::strcmp(TestType,rpc_dqm::qTestType::YRangeContent.c_str())) this->SetContentsYRangeROOTTest(mui,TestName,WarningLevel,params);    



}

std::map< std::string, std::vector<std::string> > RPCQualityTester::CheckTestsSingle(MonitorUserInterface * mui){
 
 detailedWarnings.clear();
 this->SearchDirectories(mui);
 return detailedWarnings;

} 

void RPCQualityTester::SearchDirectories(MonitorUserInterface * mui) 
{
 std::vector<std::string> meNames=mui->getMEs();   
 std::vector<std::string> dirNames=mui->getSubdirs();
 int pippo=meNames.size();
 std::string currentDir=mui->pwd();
 if(printout) {
 	logFile << "Searching ME's with quality tests in " << currentDir<<"\n"
   		       << "There are " << pippo <<" monitoring elements and "
  	  	       << dirNames.size()<<" directories\n"<< std::endl;
 }


  if(meNames.size()) {

  	if(printout) logFile<<"chiamo dir"<<std::endl;
  	this->ProcessAlarms(meNames, currentDir, mui);
  }
 
  if(dirNames.size()){
  	for(std::vector<std::string>::iterator it = dirNames.begin(); 
			it != dirNames.end();++it)
    	{
      		mui->cd(*it);
      		this->SearchDirectories(mui);
    	}   
   }   	
	
  mui->goUp();
  
  return;
}


void RPCQualityTester::ProcessAlarms(std::vector<std::string> meNames, std::string currentDir, MonitorUserInterface * mui){
    logFile<<"sonoqui0, number of me's: "<<meNames.size()<<std::endl;

 for(std::vector<std::string>::iterator nameItr= meNames.begin(); nameItr!= meNames.end(); ++nameItr){
      
      std::string colour;
      char text[128];
        
	MonitorElement * me =0;
	logFile<<"sonoquinome "<<*nameItr<<std::endl;
	char fullPath[128];
	sprintf(fullPath,"%s/%s",currentDir.c_str(),(*nameItr).c_str());
	logFile<<"fullpath "<<fullPath<<std::endl;
	me= mui->get(fullPath);
        logFile<<"sonoqui1 "<<me<<std::endl;
	std::vector<QReport *> report;
	if(me){
		 if (me->hasError()){
		 	colour="red";
			report= me->getQErrors();
        		logFile<<"sonoqui1 "<<std::endl;
		 } 
		 if( me->hasWarning()){ 
		 	colour="orange";
			report= me->getQWarnings();
        		logFile<<"sonoqui2 "<< std::endl;
		 }
		 if(me->hasOtherReport()){
		 	colour="black";
 			report= me->getQOthers();
       			logFile<<"sonoqui3 "<<std::endl;
		 }
	
		 for(std::vector<QReport *>::iterator itr=report.begin(); itr!=report.end();++itr ){
		        sprintf(text,"%s:%s",(*nameItr).c_str(),((*itr)->getMessage()).c_str());
			logFile<<"report "<<(*itr)->getMessage()<<std::endl;
			logFile<<"text= "<<text<<std::endl;
	    		
			std::vector<std::string> messageList;
	    		if( detailedWarnings.find(colour) == detailedWarnings.end()){
 	      			messageList.push_back(text);
	      			detailedWarnings[colour]=messageList;
	    		}else{
	      			messageList=detailedWarnings[colour];
	      			messageList.push_back(text);
	    		}	    
		 }	
	}
 }

}

/** \file
 *
 *  Implementation of RPCQualityTester
 *
 *  $Date: 2006/01/25 16:28:39 $
 *  $Revision: 1.1 $
 *  \author Ilaria Segoni
 */
#include "DQM/RPCMonitorClient/interface/RPCQualityTester.h"
#include "DQMServices/QualityTests/interface/QCriterionRoot.h"

#include<iostream>


RPCQualityTester::RPCQualityTester()
{
  
  printout=true;
  logFile.open("RPCDQMClient.log");
  
  if(printout) logFile <<"An RPC Quality Tester Is being Created"<<std::endl;

}


RPCQualityTester::~RPCQualityTester(){ 
  logFile.close();
}



void RPCQualityTester::SetupTestsFromDB(MonitorUserInterface * mui){ 


  if(printout) logFile<<"in RPCQualityTester::SetupTestsFromDB"<<std::endl;
  FILE* TestsFile;
  char  fileName[128];
  sprintf(fileName,"QualityTests.db");


  
 TestsFile= fopen(fileName,"r");
  
  
  int TestType;
  char TestName[20];
  float WarningLevel=0;
  float params[5];
  
  for(int ii=0; ii<5; ii++) params[ii]=0.0;


  int counter =1;
  int file_status=1;
  
  do{
    
    file_status=fscanf(TestsFile,"%d%s%f%f%f%f%f%f\n",
                         &TestType,&TestName,&WarningLevel,
			 &params[0],&params[1],&params[2],&params[3],&params[4]);
			 
    if(file_status != EOF){			 
       if(printout)  logFile<<" Reading Quality Tests Configuration: "
            <<TestType<<" "<<TestName<<" "<<WarningLevel<<" "
	    <<params[0]<<" "<<params[1]<<" "<<params[2]
	    <<" "<<params[3]<<" "<<params[4]<< std::endl;
	
     if(TestType==1) this->SetContentsXRangeROOTTest(mui,TestName,WarningLevel,params);    
       	    
    }	

     counter++;


 }while( file_status!=EOF ); 



}



void RPCQualityTester::SetContentsXRangeROOTTest(MonitorUserInterface * mui, char  TestName[20], float  WarningLevel, float  params[5]){

  if(printout) logFile<<"in RPCQualityTester::SetContentsXRangeROOTTest"<<std::endl;
	
 /// X-axis content withing a given range [Xmin, Xmax]
 qTests.push_back(TestName);
 
 
 QCriterion * qc1 = mui->createQTest(ContentsXRangeROOT::getAlgoName(),TestName);
 MEContentsXRangeROOT * me_qc1 = (MEContentsXRangeROOT *) qc1;
 me_qc1->setAllowedXRange(params[0],params[1]);
 me_qc1->setWarningProb(WarningLevel);
 
}



void RPCQualityTester::SetupTests(MonitorUserInterface * mui){    
  
  this-> SetupTestsFromDB(mui); 
 
 /// X-axis content withing a given range [Xmin, Xmax]
 //qtest1 = "xRange";
 //QCriterion * qc1 = mui->createQTest(ContentsXRangeROOT::getAlgoName(),qtest1);
 //MEContentsXRangeROOT * me_qc1 = (MEContentsXRangeROOT *) qc1;
 /// set allowed range in X-axis (default values: histogram's X-range)
 //me_qc1->setAllowedXRange(1.0, 200.0);
 /// set probability limit for test warning (default: 90%)
 //me_qc1->setWarningProb(0.90);
 return;

}

void RPCQualityTester::AttachTests(MonitorUserInterface * mui){

 /// use test <qtest1> on all MEs matching string ""
 if(printout) logFile<<"NumberOfChannelsWithData ..."<<std::endl;
 mui->useQTest("NumberOfChannelsWithData", qtest1);
 if(printout) logFile<<"...done."<<std::endl;


}



void RPCQualityTester::CheckTests(MonitorUserInterface * mui) 
{
  if(printout) logFile << "Checking Status of Quality Tests" << std::endl;

  
	int status = 1980;
	
	status= mui->getSystemStatus();
        if(printout) logFile << "The STATUS IS " << status<<std::endl;
        if(printout) logFile << "Possible states: \nerror:  " 
	<< dqm::qstatus::ERROR<<
	"\nwarning:  "<< dqm::qstatus::WARNING<<
	"\n Other: "<< dqm::qstatus::OTHER<<std::endl;
	
	switch(status)
	  {
	  case dqm::qstatus::ERROR:
	    std::cout << " Error(s)";
	    break;
	  case dqm::qstatus::WARNING:
	    std::cout << " Warning(s)";
	    break;
	  case dqm::qstatus::OTHER:
	    std::cout << " Some tests did not run;";
	    break; 
	  default:
	    std::cout << " No problems";
	  }

	//MonitorElement * me4 = mui->get("Collector/FU0/C1/C2/histo4");
	//if(me4)
	  //checkTests(me4);
	
  return;
}

  

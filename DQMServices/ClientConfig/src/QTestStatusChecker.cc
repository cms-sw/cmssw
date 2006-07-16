/** \file
 *
 *  Implementation of QTestStatusChecker
 *
 *  $Date: 2006/07/13 16:36:23 $
 *  $Revision: 1.3 $
 *  \author Ilaria Segoni
 */

#include "DQMServices/ClientConfig/interface/QTestStatusChecker.h"
#include <iostream>

QTestStatusChecker::QTestStatusChecker(){

}

QTestStatusChecker::~QTestStatusChecker(){
}

std::pair<std::string,std::string> QTestStatusChecker::checkGlobalStatus(MonitorUserInterface * mui){
	std::pair<std::string,std::string> statement;
	int status= mui->getSystemStatus();
	switch(status){
		case dqm::qstatus::ERROR:
			statement.first ="Errors detected in quality tests";
			statement.second="red";
			break;
	  	case dqm::qstatus::WARNING:
			statement.first ="Warnings detected in quality tests";
			statement.second="orange";
			break;
	 	case dqm::qstatus::OTHER:
	    		statement.first="Some tests did not run";
	    		statement.second="black";
	    		break; 
	  	default:
	    		statement.first="No problems detected in quality tests ";
	    		statement.second="green";
	}

	return statement;
}

std::map< std::string, std::vector<std::string> > QTestStatusChecker::checkDetailedStatus(MonitorUserInterface * mui){ 
	
	std::vector<std::string> allPathNames=this->fullPathNames(mui); 
	detailedWarnings.clear();

	this->processAlarms(allPathNames,mui);	
	return detailedWarnings;
} 

		
void QTestStatusChecker::processAlarms(std::vector<std::string> allPathNames, MonitorUserInterface * mui){	
  
 for(std::vector<std::string>::iterator fullMePath=allPathNames.begin();fullMePath!=allPathNames.end(); ++fullMePath ){		
        
        MonitorElement * me=0;	
        me= mui->get(*fullMePath);

	if(me){
		std::vector<QReport *> report;
        	std::string colour;

		if (me->hasError()){
			colour="red";
			report= me->getQErrors();
 		 } 
 		 if( me->hasWarning()){ 
 			 colour="orange";
 			 report= me->getQWarnings();
 		 }
 		 if(me->hasOtherReport()){
			 colour="black";
 			 report= me->getQOthers();
 		 }
		 for(std::vector<QReport *>::iterator itr=report.begin(); itr!=report.end();++itr ){
			 std::string text= (*fullMePath) + (*itr)->getMessage();
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


std::vector<std::string> QTestStatusChecker::fullPathNames(MonitorUserInterface * mui){


  std::vector<std::string> contents;
  std::vector<std::string> contentVec;
  mui->getContents(contentVec);
  for (std::vector<std::string>::iterator it = contentVec.begin();
       it != contentVec.end(); it++) {
        
	std::string::size_type dirCharNumber = it->find( ":", 0 );
	std::string dirName=it->substr(0 , dirCharNumber);
	dirName+= "/"; 
	std::string meCollectionName=it->substr(dirCharNumber+1);
    
	std::string reminingNames=meCollectionName;
	bool anotherME=true;
	while(anotherME){
		if(reminingNames.find(",") == std::string::npos) anotherME =false;
		std::string::size_type singleMeNameCharNumber= reminingNames.find( ",", 0 );
    		std::string singleMeName=reminingNames.substr(0 , singleMeNameCharNumber );
        	std::string fullpath=dirName + singleMeName;
		contents.push_back(fullpath);        
		reminingNames=reminingNames.substr(singleMeNameCharNumber+1);    
	}
  }
	
  return contents;	

}

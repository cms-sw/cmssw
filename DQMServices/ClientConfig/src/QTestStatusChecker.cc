/** \file
 *
 *  Implementation of QTestStatusChecker
 *
 *  $Date: 2006/05/09 21:28:37 $
 *  $Revision: 1.1 $
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

			std::cout<<"In checkglobal"<<statement.first<<std::endl;
	return statement;
}

std::map< std::string, std::vector<std::string> > QTestStatusChecker::checkDetailedStatus(MonitorUserInterface * mui){ 
	detailedWarnings.clear();
	return detailedWarnings;
} 



/** \file
 *
 *  Implementation of  SubscriptionHandle
 *
 *  $Date: 2006/05/04 10:27:20 $
 *  $Revision: 1.1 $
 *  \author Ilaria Segoni
 */

#include "DQMServices/ClientConfig/interface/SubscriptionHandle.h"
#include "DQMServices/ClientConfig/interface/MESubscriptionParser.h"

SubscriptionHandle::SubscriptionHandle(){
	meListParser = new  MESubscriptionParser();
	subList.clear();
	unsubList.clear();
}

SubscriptionHandle::~SubscriptionHandle(){
	delete meListParser;
	meListParser=0;
}

bool SubscriptionHandle::getMEList(std::string meListFile){
	meListParser->getDocument(meListFile);
	
	if(meListParser->parseMESubscription()){
		return true;
	}else{		
		subList=meListParser->subscribeList();
		unsubList=meListParser->unsubscribeList();
		return  false;
	}
}

void SubscriptionHandle::makeSubscriptions(MonitorUserInterface * mui){	
	
	for(std::vector<std::string>::iterator meUnsubItr=unsubList.begin();
		 meUnsubItr!=unsubList.end();++meUnsubItr){
			mui->unsubscribe(*meUnsubItr);		 
	}

	for(std::vector<std::string>::iterator meSubItr=subList.begin();
		 meSubItr!=subList.end();++meSubItr){
			mui->subscribe(*meSubItr);		 
	}

}

void SubscriptionHandle::updateSubscriptions(MonitorUserInterface * mui){	
	
	for(std::vector<std::string>::iterator meSubItr=subList.begin();
		 meSubItr!=subList.end();++meSubItr){
			mui->subscribeNew(*meSubItr);		 
	}

}

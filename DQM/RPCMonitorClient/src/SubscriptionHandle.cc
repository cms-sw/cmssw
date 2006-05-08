/** \file
 *
 *  Implementation of  SubscriptionHandle
 *
 *  $Date: 2006/05/04 10:27:20 $
 *  $Revision: 1.1 $
 *  \author Ilaria Segoni
 */

#include "DQM/RPCMonitorClient/interface/SubscriptionHandle.h"
#include "DQM/RPCMonitorClient/interface/MESubscriptionParser.h"

SubscriptionHandle::SubscriptionHandle(){
	meListParser = new  MESubscriptionParser();
	subList.clear();
	unsubList.clear();
}

SubscriptionHandle::~SubscriptionHandle(){
	delete meListParser;
	meListParser=0;
}

bool SubscriptionHandle::configure(std::string meListFile){
	meListParser->getDocument(meListFile);
	subList=meListParser->subscribeList();
	unsubList=meListParser->unsubscribeList();
	return  meListParser->parseMESubscription();
}

void SubscriptionHandle::enable(MonitorUserInterface * mui){	
	
	for(std::vector<std::string>::iterator meUnsubItr=unsubList.begin();
		 meUnsubItr!=subList.end();++meUnsubItr){
			mui->unsubscribe(*meUnsubItr);		 
	}

	for(std::vector<std::string>::iterator meSubItr=subList.begin();
		 meSubItr!=subList.end();++meSubItr){
			mui->subscribe(*meSubItr);		 
	}

}

void SubscriptionHandle::onUpdate(MonitorUserInterface * mui){	
	
	for(std::vector<std::string>::iterator meSubItr=subList.begin();
		 meSubItr!=subList.end();++meSubItr){
			mui->subscribeNew(*meSubItr);		 
	}

}

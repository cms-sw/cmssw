/** \file
 *
 *  Implementation of  SubscriptionHandle
 *
 *  $Date: 2006/04/24 10:04:08 $
 *  $Revision: 1.3 $
 *  \author Ilaria Segoni
 */

#include "DQM/RPCMonitorClient/interface/SubscriptionHandle.h"

SubscriptionHandle::SubscriptionHandle(){
}

SubscriptionHandle::~SubscriptionHandle(){

}

void SubscriptionHandle::setSubscriptions(std::vector<std::string> meSubscribe, 
	std::vector<std::string> meUnubscribe, MonitorUserInterface *mui){
	
	
	for(std::vector<std::string>::iterator meUnsubItr=meUnubscribe.begin();
		 meUnsubItr!=meUnubscribe.end();++meUnsubItr){
			mui->unsubscribe(*meUnsubItr);		 
	}

	for(std::vector<std::string>::iterator meSubItr=meSubscribe.begin();
		 meSubItr!=meSubscribe.end();++meSubItr){
			mui->subscribeNew(*meSubItr);		 
	}

}

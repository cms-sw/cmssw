#ifndef SubscriptionHandle_H
#define SubscriptionHandle_H

/** \class SubscriptionHandle  
 * *
 *  using a MonitorUserInterface pointer, subscribes and unsubscribes to
 *  MonitorElements that are in the two lists
 *  std::vector<std::string> meSubscribe and
 *  std::vector<std::string> meUnubscribe;
 *
 *  $Date: 2006/04/24 09:54:22 $
 *  $Revision: 1.3 $
 *  \author Ilaria Segoni
  */

#include "DQMServices/UI/interface/MonitorUIRoot.h"

#include<vector>
#include<string>

class SubscriptionHandle{

 public:
	///Creator
	SubscriptionHandle();
	///Destructor
	~SubscriptionHandle();
	
	/// Performs the subscription
	void setSubscriptions(std::vector<std::string> meSubscribe, std::vector<std::string> meUnubscribe, MonitorUserInterface *mui);

 
 private:


};

#endif

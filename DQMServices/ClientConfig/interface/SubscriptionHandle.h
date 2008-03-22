#ifndef SubscriptionHandle_H
#define SubscriptionHandle_H

/** \class SubscriptionHandle  
 * *
 *  Controls the parsing of the XML configuration file with the list of 
 *  monitorElement to be (un)subscrubed to and the (un)subscriptions.
 *
 *  $Date: 2006/05/04 10:27:17 $
 *  $Revision: 1.1 $
 *  \author Ilaria Segoni
  */

#include "DQMServices/UI/interface/MonitorUIRoot.h"

#include<vector>
#include<string>

class MESubscriptionParser;

class SubscriptionHandle{

 public:
	///Creator
	SubscriptionHandle();
	///Destructor
	~SubscriptionHandle();
	///Parses the xml file
	bool getMEList(std::string meListFile);
	/// Performs the subscription
	void makeSubscriptions(MonitorUserInterface * mui);
 	/// Performs the subscription
	void updateSubscriptions(MonitorUserInterface * mui);

 private:

	MESubscriptionParser * meListParser;
	std::vector<std::string> subList;
	std::vector<std::string> unsubList;
};

#endif

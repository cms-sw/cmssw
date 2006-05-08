#ifndef SubscriptionHandle_H
#define SubscriptionHandle_H

/** \class SubscriptionHandle  
 * *
 *  using a MonitorUserInterface pointer, subscribes and unsubscribes to
 *  MonitorElements that are in the two lists
 *  std::vector<std::string> meSubscribe and
 *  std::vector<std::string> meUnubscribe;
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
	bool configure(std::string meListFile);
	/// Performs the subscription
	void enable(MonitorUserInterface * mui);
 	/// Performs the subscription
	void onUpdate(MonitorUserInterface * mui);

 private:

	MESubscriptionParser * meListParser;
	std::vector<std::string> subList;
	std::vector<std::string> unsubList;
};

#endif

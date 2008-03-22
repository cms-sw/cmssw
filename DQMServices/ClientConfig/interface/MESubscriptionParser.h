#ifndef MESubscriptionParser_H
#define MESubscriptionParser_H

/** \class MESubscriptionParser
 * *
 *  Parses the xml file with the configuration of quality tests
 *  and the map between quality tests and MonitorElement
 * 
 *  $Date: 2006/05/04 10:26:55 $
 *  $Revision: 1.1 $
 *  \author Ilaria Segoni
  */

#include "DQMServices/ClientConfig/interface/DQMParserBase.h"

#include<iostream>
#include<string>
#include<vector>
#include<map>

class MESubscriptionParser : public DQMParserBase {

 public:
	 ///Creator
	 MESubscriptionParser();
	 ///Destructor
	 ~MESubscriptionParser();
	 ///Method that parsesdrives the parsing of the xml file configFile, returns false if no errors are encountered
	 bool parseMESubscription();
	 /// Returns the list of ME's to subscribed to
	 std::vector<std::string> subscribeList() const { return meSubscribe;}		
	 ///  Returns the list of ME's to unsubscribe to
	 std::vector<std::string> unsubscribeList() const { return meUnubscribe;}		
	
 private:	 
	 bool parseFile();
	 
 private:	 
	 static int n_Instances;
	 std::vector<std::string> meSubscribe;		
	 std::vector<std::string> meUnubscribe;		
	 	 


};


#endif

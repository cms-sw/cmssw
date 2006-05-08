#ifndef QTestHandle_H
#define QTestHandle_H

/** \class QTestHandle
 * *
 *  Handles quality tests (configuring, attaching to ME's, 
 *  subscribing to the ME's for which quality tests are requested).
 *
 *  $Date: 2006/05/04 10:27:22 $
 *  $Revision: 1.2 $
 *  \author Ilaria Segoni
  */
  
#include<string>

class MonitorUserInterface;
class QTestConfigurationParser;
class QTestConfigure;
class QTestEnabler;
class QTestStatusChecker;

class QTestHandle{
  public:
	///Creator
	QTestHandle();
	///Destructor
	~QTestHandle();
	///Parses Config File
	bool configure(std::string configFile, MonitorUserInterface * mui);
	///...
	void enable(MonitorUserInterface * mui);
	///Possible actions to be executed on update
	enum onUpdateAction {
		checkGlobal=1,
		checkDetailed=2
	};
	///..
	void onUpdate(onUpdateAction action,MonitorUserInterface * mui);
	
	void checkGolbalQTStatus(MonitorUserInterface * mui) const;
	void checkDetailedQTStatus(MonitorUserInterface * mui) const;
  
  private:

	QTestConfigurationParser * qtParser;
	QTestConfigure * qtConfigurer;
	QTestEnabler * qtEnabler;
	QTestStatusChecker * qtChecker;


};


#endif

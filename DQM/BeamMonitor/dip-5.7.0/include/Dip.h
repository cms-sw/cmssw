#ifndef DIP_H_INCLUDED
#define DIP_H_INCLUDED


#include "Options.h"
#include "DipException.h"
#include "StdTypes.h"
#include "DipSubscriptionListener.h"
#include "DipPublicationErrorHandler.h"
#include "DipFactory.h"

#include "DipDimErrorHandler.h"

//Forward declarations.
class DipErrorHandler;
class ExitHandler;

/**
* This is used to set up the DIP infrastructure.
* Singleton
*/
class DipDllExp Dip {

	friend class DipErrorHandler;

private:
	/**
	* DO not allow copying
	*/
	Dip(const Dip& other);
	
	/**
	* DO not allow copying
	*/
	Dip& operator=(const Dip& other);

protected:
	// single instance per process
	static DipFactory* theFactory;

	static DipDimErrorHandler* handler;
	
//	static boost::property_tree::ptree m_diagnosticPTree;

public:

	/**
	* Create a DipFactory Object - Dip can act as a publisher and subscriber
	* @param name - name of the publisher (will not appear in a publication name) - (call retains ownership of memory)
	* @param logPropertiesFilePath - full file system access path to a logging properties file (defaults to current working directory)
	* @param loggingConfigWatchInterval - interval in seconds at which to inspect the logging properties file for changes (e.g. change in logging level, addition of new loggers etc...)
	                            - A value of -1 (default) indicates that the file is not watched, but only loaded at startup time.
	* @returns ptr to factory - NOT OWNED BY CALLER
	* @throw DipInternalError - if unable to set up the DIP infra structure / unexpected exception
	*/
	static DipFactory* create(const char *name, const char *logPropertiesFilePath = "log4cplus.properties", const int loggingConfigWatchInterval = -1);

	/**
	* Register a DIM error handler to handle low-level DIM issues
	* @param errHandler  A DipDimErrorHandler implementation that handles incoming error reports from DIM.
	*/
	static void addDipDimErrorHandler(DipDimErrorHandler * errHandler);
	
	/**
	* Create a DipFactory Object - Dip will act as a subscriber
	* @returns ptr to factory - NOT OWNED BY CALLER
	* @throw DipInternalError - if unable to set up the DIP infra structure / unexpected exception
	*/
	static DipFactory* create();

	/**
	* Shut DIP down - not DIP objects may be used after this
	* @throw DipInternalError unexpected exception
	*/
	static void shutdown();
};

#endif //DIP_H_INCLUDED

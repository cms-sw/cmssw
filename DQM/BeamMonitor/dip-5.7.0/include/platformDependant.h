// platformDependant.h: interface for the CplatformDependant class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_PLATFORMDEPENDANT_H__C56D650D_E33C_4AB2_AFF7_D60A812FC056__INCLUDED_)
#define AFX_PLATFORMDEPENDANT_H__C56D650D_E33C_4AB2_AFF7_D60A812FC056__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000


#include "platformDependantOptions.h"
#ifdef WIN32
#include "Windows.h"
#else
#include <pthread.h>
#include <unistd.h>
#include <sys/time.h>
#endif
#include <string>
#include <iostream>
#include "SString.h"



#ifdef WIN32
#define CDIRSEP "\\"
#else
#define CDIRSEP "//"
#endif

class PLATFORMDEPENDANT_DLL_API SThread{
private:
#ifdef WIN32
	/// Used to start threadMain()
	static void ThreadLaunch(LPVOID lpParam);
	/// holds the handle of the thread
	int hThread; 
	
	DWORD dwThreadId;
#else
	pthread_t hThread;
	static void * ThreadLaunch(void* lpParam);
#endif


protected:
	virtual void threadMain() = 0;


	bool shouldStop;
	
public:
	SThread();

	void start();

	void stop();

	bool isRunning();

	static void msSleep(unsigned long duration_ms);
};




/**
* This is a wrapper for a recursive mutex
*/
class PLATFORMDEPENDANT_DLL_API SMutex{
	/**
	* Counts the number of times the same thread has locked this mutex
	*/
	int lockCount;

#ifdef WIN32
  HANDLE hMutex;
#else
	pthread_mutex_t hMutex;
#endif

	SString mutexName;

public:
	SMutex(const SString & theName);

	virtual ~SMutex();

	/**
	* will return when holds lock
	*/
	void lock();

	// does nothing if lock not held.
	void release();

	const SString & getName();

	// returned num times the same thread has locked the mutex.
	int getLockCount();
};


/// releases associated mutex as soon as instance goes out of scope
class PLATFORMDEPENDANT_DLL_API SmutexGuard{
	SMutex & mux;
public:
	SmutexGuard(SMutex &m);

	virtual ~SmutexGuard();
};



class PLATFORMDEPENDANT_DLL_API SEvent{
private:
#ifdef WIN32
	HANDLE event;
#else 
	pthread_mutex_t hMutex;
	pthread_cond_t  hCondition;
	bool eventPending;
#endif

	SString eventName;
public:
	SEvent(const SString &name);


	/**
	* Clear the event if it is pending.
	*/
	void clearPending();

	/**
	* Blocks untill the event occurs
	* @param timeOutMsec <0 means forever, 
	* @returns 0 OK, -1 timeout, -2 waitfailed.
	*/
	short wait(long timeOutMsec);

	/**
	* Signal the occurence of the event this obejct represents.
	*/
	void signal();
};



class PLATFORMDEPENDANT_DLL_API CSharedLibrary{
private:
#ifdef WIN32
	HINSTANCE hLibrary;
#else
	void * hLibrary;
#endif

	SString fullName;

public:
	CSharedLibrary(SString & fullyQualifiedLibName);

	const SString & getFullName() const;


	void * getFunction(SString & decoratedFunctionName);



	~CSharedLibrary();
};

#endif // !defined(AFX_PLATFORMDEPENDANT_H__C56D650D_E33C_4AB2_AFF7_D60A812FC056__INCLUDED_)

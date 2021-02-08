#ifndef OPTIONS_H
#define OPTIONS_H


#define DIMSTATICLIB
#ifdef WIN32
#pragma warning(disable:4251)  // disable DLL warnings from std::string.
#pragma warning(disable:4275)  // disable DLL warnings exportation deriving std namespece base classes.
#pragma warning(disable:4290)  // disable throw specs warnings introduced in VC++ .NET 2003.
#pragma warning(disable:4786)  // disable concatinated id messages for std::container.
#pragma warning(disable:4996)  // disable BOOST deprecated (remove this when Boost version is > 1.39)
#endif

// This is to make a DIP DLL, Dip.cpp should do #define DIPLIB (CG) 
#ifdef WIN32
#if !defined(_DLL)
#if !defined(_DEBUG)
	#error Please link to the multithreaded DLL C-runtime (use the /MD switch)
#else
	#error Please link to the debug multithreaded DLL C-runtime (use the /MDd switch)
#endif
#endif


#ifdef DIPSTATICLIB
#define DipDllExp
#else
#ifdef DIPLIB
#	define DipDllExp __declspec(dllexport)
#else
#	define DipDllExp __declspec(dllimport)
#endif // DIPLIB
#endif // DIPSTATICLIB
#else  // NOT WIN32
#define DipDllExp
#endif //WIN32

#include <stdio.h>
#include <time.h>


void printTimeStamp();

#define DEBUG(level,message) printTimeStamp();printf(level);printf message
#define DEBUGLOCATION(level, message) printf(level);printf(" %s %i ",__FILE__, __LINE__);printf message
#ifdef _DEBUG
#define DTRACE(message) DEBUG("TRACE:", message)
#define DINFO(message)  DEBUG("INFO:", message)
#define DWARNING(message) DEBUGLOCATION("WARNING:", message)
#define DERROR(message) DEBUGLOCATION("ERROR:", message)
#else
#define DTRACE(message)
#define DINFO(message)
#define DWARNING(message) DEBUGLOCATION("WARNING:", message)
#define DERROR(message)	  DEBUGLOCATION("ERROR:", message)
#endif


#endif


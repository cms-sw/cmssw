#ifndef Framework_Log_h
#define Framework_Log_h

/*
	Author: Jim Kowalkowski  26-01-06

	$Id$

	This is a temporary solution to the message logger dependency problem.
	When the problem is solved, this file can point at the real message
	logger and all the calls will be changed to use it.
*/

#if 0
// should be the following, but not until it works
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#else
// temporary replacement
#include <iostream>
#define LogWarning(CAT) std::cerr << "WARNING: " << CAT << ": "
#define LogError(CAT) std::cerr << "ERROR: " << CAT << ": "
#define LogInfo(CAT) std::cerr << "INFO: " << CAT << ": "
#endif

#endif


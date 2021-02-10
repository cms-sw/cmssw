#ifndef DIPVERSION_H
#define DIPVERSION_H

#include "Options.h"
#include <assert.h>
#include <string>

class DipDllExp DipVersion{
	static const std::string dipVersion;
public:

/**
*	Get the version string from the version of DIP being used.
*/
static const std::string & getDipVersion();

/**
* Fixed len - must NEVER change, All DIMDIP type services will
* rely on this.
* Update: CG 13/09/2018 
* No longer true see note in DipVersion.cpp
*/
static unsigned int getDipVersionStringLen();
/* Get DIP Version length from data */
static unsigned int getDipVersionStringLen(const std::string & dimData);
/* temporary until all DIP Subscribers above v5.7 */
static const std::string & getPublicationDipVersion();
};

#endif

#ifndef DIPDIMERRORHANDLER_H
#define DIPDIMERRORHANDLER_H

#include "DipFactory.h"

class DipDllExp DipDimErrorHandler
{
public:
	virtual void handleException(int severity, int code, char *msg) = 0;
protected:
	~DipDimErrorHandler()  {};

};

#endif


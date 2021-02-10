#ifndef DIPUPDATEREQUESTLISTENER_H_INCLUDED
#define DIPUPDATEREQUESTLISTENER_H_INCLUDED

#include "Options.h"

class DipPublication;

class DipDllExp DipUpdateRequestListener {
public:
	virtual ~DipUpdateRequestListener() { }

	virtual void handleUpdateRequest(DipPublication* publication) = 0;
};

#endif //DIPUPDATEREQUESTLISTENER_H_INCLUDED


#ifndef DIPPUBLICATIONERRORHANDLER_H
#define DIPPUBLICATIONERRORHANDLER_H


#include "DipPublication.h"

/**
 * This interface enables asynchronous DIP errors to be reported.
 * Dip errors that can not be associated with any DIP method call are reported
 * through this interface.<p>
 * This interface must be implemented by the DIP developer.
 */
class DipDllExp DipPublicationErrorHandler{
public:
	/**
     * invoked when an asynch error occurs.
     * @param Publication source of the error
     * @param error description.
     * */
	virtual void handleException(DipPublication* publication, DipException& ex) = 0;
protected:
	~DipPublicationErrorHandler() {}
};


#endif


#ifndef DIPSUBSCRIPTION_H_INCLUDED
#define DIPSUBSCRIPTION_H_INCLUDED

#include "DipDataType.h"
#include "StdTypes.h"


/**
 * Interface for DIP subscriptions. DipSubscription objects are created by the {@link cern.dip.DIP DIP} factory.
 */
class DipDllExp DipSubscription {
public:
	/**
   * Returns Name of the DipData item this object is subscribed to.
   */
	virtual const char* getTopicName() = 0;

	/**
   * Requests last value to be re-send. This value will be received via the normal subscription
   * listener mechanism.
   */
	virtual void requestUpdate() = 0;

protected:
	DipSubscription() {}	
	virtual ~DipSubscription() {}
};

#endif //DIPSUBSCRIPTION_H_INCLUDED


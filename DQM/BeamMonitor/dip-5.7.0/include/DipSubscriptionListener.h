#ifndef DIPSUBSCRIPTIONLISTENER_H_INCLUDED
#define DIPSUBSCRIPTIONLISTENER_H_INCLUDED

class DipSubscription;
class DipData;
class DipException;


/**
 * Interface for DIP subscription Listeners. DipSubscriptionListener objects must
 * implement listener methods for reception of DIP data and exceptions. The
 * initial state of the connection to the publisher can be assumed "disconnected"
 * until connected() has been called;
 */
class DipDllExp DipSubscriptionListener {
public:
	virtual ~DipSubscriptionListener() { }

	/**
   * Will be called on reception of updated publication information. Implementation must be provided by the 
   * developer. It is important to check the quality of the data as this may determine how its value is treated. 
   * It is essential that the implementor determines the quality of the message received before processing the messages
   * data.
   * @param  subscription - the subscription object whose publication has been updated (Thus allowing one handler to be used for
   * multiple subscribtions). 
   * @param  message - contains updated publication data. The value, timestamp, data quality etc. can be extracted from the message with
   * DipData methods.  
   */
	virtual void handleMessage(DipSubscription* subscription, DipData& message) = 0;

	/**
   * Will be called when the subscription has been disconnected from the message
   * provider (Due to the publisher becoming unavailable or some failure in the 
   * DIP protocol). The developer must provide the appropriate implementation.
   * @param subscription - indicates which subscription is broken (DIP will automatically attempt to resubscribe).
   * @param reason - why the subscription broke.
   */
	virtual void disconnected(DipSubscription* subscription, char* reason) = 0;


	 /**
   * Will be called when the subscription has been (re)connected to the publication's publisher. 
   * The developer must provide the appropriate implementation.
   * @param subscription - indicates which subscription is restored/active.
   */
	virtual void connected(DipSubscription* subscription) = 0;


	 /**
   * Will be called when an exception, other than the disconnection has occured.
   * These exceptions are generally asynch and , should they occur, related to the
   * underlying DIP implementation. The exception to this is any uncaught exceptions in the
   * above handlers will be forwarded to this handler.
   * @param subscription the subscription this is causing the problem
   * @ex problem description
   */
	virtual void handleException(DipSubscription* subscription, DipException& ex) = 0;
};

#endif //DIPSUBSCRIPTIONLISTENER_H_INCLUDED


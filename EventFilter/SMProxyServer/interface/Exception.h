// $Id: Exception.h,v 1.2 2011/03/07 15:41:54 mommsen Exp $
/// @file: Exception.h 

#ifndef EventFilter_SMProxyServer_Exception_h
#define EventFilter_SMProxyServer_Exception_h


#include "xcept/Exception.h"

namespace smproxy {

  /**
     List of exceptions thrown by the SMProxyServer

     $Author: mommsen $
     $Revision: 1.2 $
     $Date: 2011/03/07 15:41:54 $
  */
}

/**
 * Generic exception raised by the storage manager proxy server
 */
XCEPT_DEFINE_EXCEPTION(smproxy, Exception)

/**
 * Exception raised in case of a finite state machine error
 */
XCEPT_DEFINE_EXCEPTION(smproxy, StateMachine)

/**
 * Exception raised in case of a monitoring error
 */
XCEPT_DEFINE_EXCEPTION(smproxy, Monitoring)

/**
 * Exception raised in case of problems accessing the info space
 */
XCEPT_DEFINE_EXCEPTION(smproxy, Infospace)

/**
 * Exception raised in case of configuration problems
 */
XCEPT_DEFINE_EXCEPTION(smproxy, Configuration)

/**
 * Exception raised when a non-existant queue is requested.
 */
XCEPT_DEFINE_EXCEPTION(smproxy, UnknownQueueId)

/**
 * Data retrieving problem
 */
XCEPT_DEFINE_EXCEPTION(smproxy, DataRetrieval)

/**
 * Consumer registration exception
 */
XCEPT_DEFINE_EXCEPTION(smproxy, ConsumerRegistration)

/**
 * DQM consumer registration exception
 */
XCEPT_DEFINE_EXCEPTION(smproxy, DQMConsumerRegistration)

/**
 * DQM archiving exception
 */
XCEPT_DEFINE_EXCEPTION(smproxy, DQMArchival)

/**
 * State transition error
 */
XCEPT_DEFINE_EXCEPTION(smproxy, StateTransition)

/**
 * Corrupted events alarm
 */
XCEPT_DEFINE_EXCEPTION(smproxy, CorruptedEvents)


#endif // EventFilter_SMProxyServer_Exception_h


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -

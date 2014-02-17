// $Id: Exception.h,v 1.16 2011/03/07 15:31:31 mommsen Exp $
/// @file: Exception.h 

#ifndef EventFilter_StorageManager_Exception_h
#define EventFilter_StorageManager_Exception_h


#include "xcept/Exception.h"


namespace stor {

  /**
     List of exceptions thrown by the StorageManager

     $Author: mommsen $
     $Revision: 1.16 $
     $Date: 2011/03/07 15:31:31 $
  */
}

/**
 * Generic exception raised by the storage manager
 */
XCEPT_DEFINE_EXCEPTION(stor, Exception)

/**
 * Exception raised in case of a SOAP error
 */
XCEPT_DEFINE_EXCEPTION(stor, SoapMessage)

/**
 * Exception raised in case of a finite state machine error
 */
XCEPT_DEFINE_EXCEPTION(stor, StateMachine)

/**
 * Exception raised in case of a monitoring error
 */
XCEPT_DEFINE_EXCEPTION(stor, Monitoring)

/**
 * Exception raised in case of problems accessing the info space
 */
XCEPT_DEFINE_EXCEPTION(stor, Infospace)

/**
 * Exception raised in case of configuration problems
 */
XCEPT_DEFINE_EXCEPTION(stor, Configuration)

/**
 * Exception raised in case of missuse of I2OChain
 */
XCEPT_DEFINE_EXCEPTION(stor, I2OChain)

/**
 * Exception raised in case of asking for information from the wrong I2O message type
 */
XCEPT_DEFINE_EXCEPTION(stor, WrongI2OMessageType)

/**
 * Alarm for events not tagged for any stream or consumer
 */
XCEPT_DEFINE_EXCEPTION(stor, UnwantedEvents)

/**
 * Alarm for too many error events
 */
XCEPT_DEFINE_EXCEPTION(stor, ErrorEvents)

/**
 * Exception raised in case of requesting information from a faulty or incomplete init message
 */
XCEPT_DEFINE_EXCEPTION(stor, IncompleteInitMessage)

/**
 * Exception raised in case of requesting information from a faulty or incomplete event message
 */
XCEPT_DEFINE_EXCEPTION(stor, IncompleteEventMessage)

/**
 * Exception raised in case of requesting information from a faulty or incomplete DQM event message
 */
XCEPT_DEFINE_EXCEPTION(stor, IncompleteDQMEventMessage)

/**
 * Exception raised if event selector cannot be initialized
 */
XCEPT_DEFINE_EXCEPTION(stor, InvalidEventSelection)

/**
 * Exception raised when the SM is unable to determine which resource
 * broker should received a discard message for a particular I2O message.
 */
XCEPT_DEFINE_EXCEPTION(stor, RBLookupFailed)

/**
 * Exception raised when a non-existant queue is requested.
 */
XCEPT_DEFINE_EXCEPTION(stor, UnknownQueueId)

/**
 * Exception raised when a non-existant stream is requested.
 */
XCEPT_DEFINE_EXCEPTION(stor, UnknownStreamId)

/**
 * Exception raised if a run number mismatch is detected.
 */
XCEPT_DEFINE_EXCEPTION(stor, RunNumberMismatch)

/**
 * Exception raised in case of a fragment processing error
 */
XCEPT_DEFINE_EXCEPTION(stor, FragmentProcessing)

/**
 * Exception raised in case of a DQM event processing error
 */
XCEPT_DEFINE_EXCEPTION(stor, DQMEventProcessing)

/**
 * Exception raised in case of a disk writing error
 */
XCEPT_DEFINE_EXCEPTION(stor, DiskWriting)

/**
 * Exception raised when an output file is truncated
 */
XCEPT_DEFINE_EXCEPTION(stor, FileTruncation)

/**
 * Exception when requested directory does not exist
 */
XCEPT_DEFINE_EXCEPTION( stor, NoSuchDirectory )

/**
 * Consumer registration exception
 */
XCEPT_DEFINE_EXCEPTION( stor, ConsumerRegistration )

/**
 * DQM consumer registration exception
 */
XCEPT_DEFINE_EXCEPTION( stor, DQMConsumerRegistration )

/**
 * Exception for sentinel alarm for CopyWorkers count
 */
XCEPT_DEFINE_EXCEPTION( stor, CopyWorkers )

/**
 * Exception for sentinel alarm for InjectWorkers count
 */
XCEPT_DEFINE_EXCEPTION( stor, InjectWorkers )

/**
 * Exception for sentinel alarm if disk space fills up
 */
XCEPT_DEFINE_EXCEPTION( stor, DiskSpaceAlarm )

/**
 * Exception for sentinel alarm if problems with SATA beasts
 */
XCEPT_DEFINE_EXCEPTION( stor, SataBeast )

/**
 * State transition error
 */
XCEPT_DEFINE_EXCEPTION( stor, StateTransition )

/**
 * Exception for sentinel alarm if faulty chains are found
 */
XCEPT_DEFINE_EXCEPTION( stor, FaultyEvents )

/**
 * Exception for sentinel alarm if discards are ignored
 */
XCEPT_DEFINE_EXCEPTION( stor, IgnoredDiscard )


#endif // EventFilter_StorageManager_Exception_h


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -

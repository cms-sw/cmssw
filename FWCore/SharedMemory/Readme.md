# FWCore/SharedMemory Documentation

## Introduction
This package contains code to assist in having two processes communicate via shared memory. The primary idea is to have one of the processes be cmsRun (with a module working as the Controller) and then having the other process do work on behalf of the module as its Worker. One module can have several Workers, for example one Worker per Stream. Although the various classes in the package are capable of working independently, when used together they handle most of the needs for interprocess communication.

The shared memory control is being handled by boost's interprocess package.

## Classes
The following are short descriptions for each class available in the package. All classes are in the edm::shared_memory namespace.

### ROOTSerializer and ROOTDeserializer
These classes do not directly use shared memory. They are wrappers around ROOT's mechanism for using dictionaries to serialize and deserialize C++ objects. These classes are used in conjunction with other classes which control the memory buffer into which the object is serialized and from which it is deserialized.

### ReadBuffer and WriteBuffer
A pair of ReadBuffer and WriteBuffer use the same shared memory area to communicate. They are capable of dynamically creating new buffers internally if there is a need to grow the buffer.

### ControllerChannel and WorkerChannel
These classes handle synchonizing communication between the Controller and Worker processes. They handle initialization of the Worker and then processing of the various framework transitions. These classes also work with the ReadBuffer and WriteBuffer objects to synchronize communication and handling the index used when the buffers change.

### WorkerMonitorThread
This class does not directly use shared memory. It is designed to use a helper thread and a signal handler to monitor the Worker process for a unix signal that will terminate the process. In the advent of such a signal, a user callable routine will be called. This is used to unlock the scoped lock being held by the WorkerChannel which will allow the Controller process to time out on its wait using that lock.

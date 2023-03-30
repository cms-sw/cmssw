# Extend CMSSW to a fully distributed application

Let multiple CMSSW processes on the same or different machines coordinate event processing and transfer data products
over MPI.

The implementation is based on four CMSSW modules.
Two are responsible for setting up the communication channels and coordinate the event processing:
  - the `MPIController`
  - the `MPISource`

and two are responsible for the transfer of data products:
  - the `MPISender`
  - the `MPIReceiver`

.

## `MPIController` class

The `MPIController` is an `EDProducer` running in a regular CMSSW process. After setting up the communication with an
`MPISource`, it transmits to it all EDM run, lumi and event transitions, and instructs the `MPISource` to replicate them
in the second process.


## `MPISource` class

The `MPISource` is a `Source` controlling the execution of a second CMSSW process. After setting up the communication
with an `MPIController`, it listens for EDM run, lumi and event transitions, and replicates them in its process.


## `MPISender` class

The `MPISender` is an `EDProducer` that can read a collection of a predefined type from the `Event`, serialise it using
its ROOT dictionary, and send it over the MPI communication channel.
`MPISender` is templated on the type to be serialised and transmitted.


## `MPIReceiver` class

The `MPIReceiver` is an `EDProducer` that can receive a collection of a predefined type over the MPI communication
channel, deserialise is using its ROOT dictionary, and put it in the `Event`.
`MPIReceiver` is templated on the type to be received and deserialised.

Both `MPISender` and `MPIReceiver` are configured with an instance value that is used to match one `MPISender` in one
process to one `MPIReceiver` in another process. Using different instance values allows the use of multiple pairs of
`MPISender`/`MPIReceiver` modules in a process.


## MPI communication channel

The `MPIController` and `MPISource` produce an MPIToken, a special data product that encapsulates the information about
the MPI communication channel.

Both `MPISender` and `MPIReceiver` obtain the MPI communication channel reading an MPIToken from the event. They also
produce a copy of the MPIToken, so other modules can consume it to declare a dependency on the previous modules.


## Testing

An automated test is available in the test/ directory.


## Current limitations

  - all communication is blocking, and there is no acknowledgment or feedback from one module to the other;
  - `MPIDriver` is a "one" module that supports only a single luminosity block at a time;
  - `MPISender` and `MPIReceiver` support a single compile-time type;
  - there is no check that the type sent by the `MPISender` matches the type expected by the `MPIReceiver`.


## Notes for future developments

  - implement efficient serialisation for standard layout types;
  - implement efficient serialisation for `PortableCollection` types;
  - check the the collection sent by the `MPISender` and the one expected by the `MPIReceiver` match;
  - extend the `MPISender` and `MPIReceiver` to send and receive multiple collections;
  - rewrite the `MPISender` and `MPIReceiver` to send and receive arbitrary run-time collections;
  - improve the `MPIController` to be a `global` module rather than a `one` module;
  - let an `MPISource` accept connections and events from multiple `MPIController` modules in different jobs;
  - let an `MPIController` connect and sent events to multiple `MPISource` modules in different jobs;
  - support multiple concurrent runs and luminosity blocks, up to a given maximum;
  - transfer the `ProcessingHistory` from the `MPIController` to the `MPISource` ? and vice-versa ?
  - transfer other provenance information from the `MPIController` to the `MPISource` ? and vice-versa ?
  - when a run, luminosity block or event is received, check that they belong to the same `ProcessingHistory` as the
    ongoing run ?

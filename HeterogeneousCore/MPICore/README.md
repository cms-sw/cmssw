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
with an `MPIController`, it listens for EDM run, lumi and event transitions, and replicates them in its own process.


## `MPISender` class

The `MPISender` is an `EDProducer` that can read any number of collections of arbitrary types from the `Event`.
For each event, it first sends a metadata message describing the products to be transferred, including their number and
basic characteristics.

If `MemoryCopyTraits` are defined for a given product, the data are transferred directly from the product's memory
regions; otherwise, the product is serialised into a single buffer using its ROOT dictionary. The regions and the buffer
are sent to another process over the MPI communication channel.

If an event has been filtered out upstream (for example via a missing `edm::PathStateToken`), the sender marks the
metadata with a negative product count so that the receiver can skip transfers for that event.

The number and types of the collections to be read from the `Event` is determined by the module configuration. The
configuration can specify a list of module labels, branch names, or a mix of the two:
  - a module label selects all collections produced by that module, irrespective of the type and instance;
  - a branch name selects only the collections that match all the branch fields (type, label, instance, process name),
    similar to an `OutputModule`'s `"keep ..."` statement.

Wildcards (`?` and `*`) are allowed in a module label or in each field of a branch name.


## `MPIReceiver` class

The `MPIReceiver` is an `EDProducer` that can receive any number of collections of arbitrary types over the MPI
communication channel. It first receives metadata, which is later used to initialise trivially copyable products and
allocate buffers for serialised products.

For trivially copyable products, the receiver initialises the target objects using the metadata and performs an
`MPI_Recv` for each memory region. For non-trivially copyable products, it receives the serialised buffer and
deserialises it using the corresponding ROOT dictionary.

All received products are put into the `Event`. The number, type and label of the collections to be produced is
determined by the module configuration.

For each collection, the `type` indicates the C++ type as understood by the ROOT dictionary, and the `label` indicates
the module instance label to be used for producing that collection into the `Event`.


## `MPISender` and `MPIReceiver` instances

Both `MPISender` and `MPIReceiver` are configured with an instance value that is used to match one `MPISender` in one
process to one `MPIReceiver` in another process. Using different instance values allows the use of multiple pairs of
`MPISender`/`MPIReceiver` modules in a process.


## MPI communication channel

The `MPIController` and `MPISource` produce an `MPIToken`, a special data product that encapsulates the information
about the MPI communication channel.

Both `MPISender` and `MPIReceiver` obtain the MPI communication channel reading an `MPIToken` from the event, identified
by the `upstream` parameter.
They also produce a copy of the `MPIToken`, so other modules can consume it to declare a dependency on those modules.


## Testing

An automated test is available in the `test/` directory.

## Automatic configuration splitter

In order to use MPI functionality in CMSSW, you need to create special configurations with MPI modules performing the 
communication. Therefore we provide ```local_remote_splitter.py``` script which allows to split any given
python process configuration into 2 parts: local and remote.

The tool takes modules to offload as a command line parameter and analyzes data dependencies between CMSSW modules.
It generates two derived configuration files: a *local* process config and a *remote* process config.

Modules specified for offloading will be moved to the remote process.
Any required data products are automatically identified and forwarded
between processes unless the producing module is explicitly marked as shared.

The example command to offload GPU component of ECAL, HCAL and Pixels would be following:

```
python3 local_remote_splitter.py hlt.py --remote-modules hltEcalDigisSoA hltEcalUncalibRecHitSoA \
        hltHcalDigisSoA hltHbheRecoSoA hltParticleFlowRecHitHBHESoA hltParticleFlowClusterHBHESoA \
        hltSiPixelClustersSoA hltSiPixelRecHitsSoA hltPixelTracksSoA hltPixelVerticesSoA \
        --duplicate-modules hltHcalDigis hltOnlineBeamSpot   hltOnlineBeamSpotDevice \
        --output-local local_pixels.py \
        --output-remote remote_pixels.py
```

For more information about input parameters of the script you could run ```local_remote_splitter.py -h```.


## Current limitations

  - `MPIController` is a "one" module that supports only a single luminosity block at a time;
  - there is only a partial check that the number, type and order of collections sent by the `MPISender` match those
    expected by the `MPIReceiver`.


## Notes for future developments

  - implement efficient GPU-direct transfers for trivially serialisable products (in progress);
  - check that the collection sent by the `MPISender` matches the one expected by the `MPIReceiver`;
  - integrate filter decisions and GPU backend into the metadata message
  - improve the `MPIController` to be a `global` module rather than a `one` module;
  - let an `MPISource` accept connections and events from multiple `MPIController` modules in different jobs;
  - let an `MPIController` connect and sent events to multiple `MPISource` modules in different jobs (in progress);
  - support multiple concurrent runs and luminosity blocks, up to a given maximum;
  - when a run, luminosity block or event is received, check that they belong to the same `ProcessingHistory` as the
    ongoing run?

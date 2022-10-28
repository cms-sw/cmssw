# SONIC for Triton Inference Server

## Introduction to Triton

Triton Inference Server ([docs](https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton_inference_server_1130/user-guide/docs/index.html), [repo](https://github.com/NVIDIA/triton-inference-server))
is an open-source product from Nvidia that facilitates the use of GPUs as a service to process inference requests.

Triton supports multiple named inputs and outputs with different types. The allowed types are:
boolean, unsigned integer (8, 16, 32, or 64 bits), integer (8, 16, 32, or 64 bits), floating point (16, 32, or 64 bit), or string.

Triton additionally supports inputs and outputs with multiple dimensions, some of which might be variable (denoted by -1).
Concrete values for variable dimensions must be specified for each call (event).

## Client

Accordingly, the `TritonClient` input and output types are:
* input: `TritonInputMap = std::unordered_map<std::string, TritonInputData>`
* output: `TritonOutputMap = std::unordered_map<std::string, TritonOutputData>`

`TritonInputData` and `TritonOutputData` are classes that store information about their relevant dimensions and types
and facilitate conversion of data sent to or received from the server.
They are stored by name in the input and output maps.
The consistency of dimension and type information (received from server vs. provided by user) is checked at runtime.
The model information from the server can be printed by enabling `verbose` output in the `TritonClient` configuration.

`TritonClient` takes several parameters:
* `modelName`: name of model with which to perform inference
* `modelVersion`: version number of model (default: -1, use latest available version on server)
* `modelConfigPath`: path to `config.pbtxt` file for the model (using `edm::FileInPath`)
* `preferredServer`: name of preferred server, for testing (see [Services](#services) below)
* `timeout`: maximum allowed time for a request (disabled with 0)
* `outputs`: optional, specify which output(s) the server should send
* `verbose`: enable verbose printouts (default: false)
* `useSharedMemory`: enable use of shared memory (see [below](#shared-memory)) with local servers (default: true)
* `compression`: enable compression of input and output data to reduce bandwidth (using gzip or deflate) (default: none)

The batch size should be set using the client accessor, in order to ensure a consistent value across all inputs:
* `setBatchSize()`: set a new batch size
  * some models may not support batching

Useful `TritonData` accessors include:
* `variableDims()`: return true if any variable dimensions
* `sizeDims()`: return product of dimensions (-1 if any variable dimensions)
* `shape()`: return actual shape (list of dimensions)
* `sizeShape()`: return product of shape dimensions (returns `sizeDims()` if no variable dimensions)
* `byteSize()`: return number of bytes for data type
* `dname()`: return name of data type
* `batchSize()`: return current batch size

To update the `TritonData` shape in the variable-dimension case:
* `setShape(const std::vector<int64_t>& newShape)`: update all (variable) dimensions with values provided in `newShape`
* `setShape(unsigned loc, int64_t val)`: update variable dimension at `loc` with `val`

There are specific local input and output containers that should be used in producers.
Here, `T` is a primitive type, and the two aliases listed below are passed to `TritonInputData::toServer()`
and returned by `TritonOutputData::fromServer()`, respectively:
* `TritonInputContainer<T> = std::shared_ptr<TritonInput<T>> = std::shared_ptr<std::vector<std::vector<T>>>`
* `TritonOutput<T> = std::vector<edm::Span<const T*>>`

The `TritonInputContainer` object should be created using the helper function described below.
It expects one vector per batch entry (i.e. the size of the outer vector is the batch size).
Therefore, it is best to call `TritonClient::setBatchSize()`, if necessary, before calling the helper.
It will also reserve the expected size of the input in each inner vector (by default),
if the concrete shape is available (i.e. `setShape()` was already called, if the input has variable dimensions).
* `allocate<T>()`: return a `TritonInputContainer` properly allocated for the batch and input sizes

### Shared memory

If the local fallback server (see [Services](#services) below) is in use,
input and output data can be transferred via shared memory rather than gRPC.
Both CPU and GPU (CUDA) shared memory are supported.
This is more efficient for some algorithms;
if shared memory is not more efficient for an algorithm, it can be disabled in the Python configuration for the client.

For outputs, shared memory can only be used if the batch size and concrete shape are known in advance,
because the shared memory region for the output must be registered before the inference call is made.
As with the inputs, this is handled automatically, and the use of shared memory can be disabled if desired.

## Modules

SONIC Triton supports producers, filters, and analyzers.
New modules should inherit from `TritonEDProducer`, `TritonEDFilter`, or `TritonOneEDAnalyzer`.
These follow essentially the same patterns described in [SonicCore](../SonicCore#for-analyzers).

If an `edm::GlobalCache` of type `T` is needed, there are two changes:
* The new module should inherit from `TritonEDProducerT<T>` or `TritonEDFilterT<T>`
* The new module should contain these lines:
    ```cpp
    static std::unique_ptr<T> initializeGlobalCache(edm::ParameterSet const& pset) {
      TritonEDProducerT<T>::initializeGlobalCache(pset);
      [module-specific code goes here]
    }
    ```

For `TritonEDProducer` and `TritonEDFilter`, the function `tritonEndStream()` replaces the standard `endStream()`.
For `TritonOneEDAnalyzer`, the function `tritonEndJob()` replaces the standard `endJob()`.

In a SONIC Triton producer, the basic flow should follow this pattern:
1. `acquire()`:  
    a. access input object(s) from `TritonInputMap`  
    b. allocate input data using `allocate<T>()`  
    c. fill input data  
    d. set input shape(s) (optional, only if any variable dimensions)  
    e. convert using `toServer()` function of input object(s)  
2. `produce()`:  
    a. access output object(s) from `TritonOutputMap`  
    b. obtain output data as `TritonOutput<T>` using `fromServer()` function of output object(s) (sets output shape(s) if variable dimensions exist)  
    c. fill output products  

## Services

A script [`cmsTriton`](./scripts/cmsTriton) is provided to launch and manage local servers.
The script has two operations (`start` and `stop`) and the following options:
* `-c`: don't cleanup temporary dir (for debugging)
* `-D`: dry run: print container commands rather than executing them
* `-d`: use Docker instead of Singularity
* `-f`: force reuse of (possibly) existing container instance
* `-g`: use GPU instead of CPU
* `-i` [name]`: server image name (default: fastml/triton-torchgeo:20.09-py3-geometric)
* `-M [dir]`: model repository (can be given more than once)
* `-m [dir]`: specific model directory (can be given more than one)
* `-n [name]`: name of container instance, also used for hidden temporary dir (default: triton_server_instance)
* `-P [port]`: base port number for services (-1: automatically find an unused port range) (default: 8000)
* `-p [pid]`: automatically shut down server when process w/ specified PID ends (-1: use parent process PID)
* `-r [num]`: number of retries when starting container (default: 3)
* `-s [dir]`: Singularity sandbox directory (default: /cvmfs/unpacked.cern.ch/registry.hub.docker.com/fastml/triton-torchgeo:20.09-py3-geometric)
* `-t [dir]`: non-default hidden temporary dir
* `-v`: (verbose) start: activate server debugging info; stop: keep server logs
* `-w [time]`: maximum time to wait for server to start (default: 300 seconds)
* `-h`: print help message and exit

Additional details and caveats:
* The `start` and `stop` operations for a given container instance should always be executed in the same directory
if a relative path is used for the hidden temporary directory (including the default from the container instance name),
in order to ensure that everything is properly cleaned up.
* A model repository is a folder that contains multiple model directories, while a model directory contains the files for a specific file.
(In the example below, `$CMSSW_BASE/src/HeterogeneousCore/SonicTriton/data/models` is a model repository,
while `$CMSSW_BASE/src/HeterogeneousCore/SonicTriton/data/models/resnet50_netdef` is a model directory.)
If a model repository is provided, all of the models it contains will be provided to the server.
* Older versions of Singularity have a short timeout that may cause launching the server to fail the first time the command is executed.
The `-r` (retry) flag exists to work around this issue.

A central `TritonService` is provided to keep track of all available servers and which models they can serve.
The servers will automatically be assigned to clients at startup.
If some models are not served by any server, the `TritonService` can launch a fallback server using the `cmsTriton` script described above.
If the process modifiers `enableSonicTriton` or `allSonicTriton` are activated,
the fallback server will launch automatically if needed and will use a local GPU if one is available.
If the fallback server uses CPU, clients that use the fallback server will automatically be set to `Sync` mode.

Servers have several available parameters:
* `name`: unique identifier for each server (clients use this when specifying preferred server; also used internally by `TritonService`)
* `address`: web address for server
* `port`: port number for server (Triton server provides gRPC service on port 8001 by default)
* `useSsl`: connect to server via SSL (default: false)
* `rootCertificates`: for SSL, name of file containing PEM encoding of server root certificates, if any
* `privateKey`: for SSL, name of file containing PEM encoding of user's private key, if any
* `certificateChain`: for SSL, name of file containing PEM encoding of user's certificate chain, if any

The fallback server has a separate set of options, mostly related to the invocation of `cmsTriton`:
* `enable`: enable the fallback server
* `debug`: enable debugging (equivalent to `-c` in `cmsTriton`)
* `verbose`: enable verbose output in logs (equivalent to `-v` in `cmsTriton`)
* `useDocker`: use Docker instead of Singularity (equivalent to `-d` in `cmsTriton`)
* `useGPU`: run on local GPU (equivalent to `-g` in `cmsTriton`)
* `retries`: number of retries when starting container (passed to `-r [num]` in `cmsTriton` if >= 0; default: -1)
* `wait`: maximum time to wait for server to start (passed to `-w time` in `cmsTriton` if >= 0; default: -1)
* `instanceBaseName`: base name for server instance if random names are enabled (default: triton_server_instance)
* `instanceName`: specific name for server instance as alternative to random name (passed to `-n [name]` in `cmsTriton` if non-empty)
* `tempDir`: specific name for server temporary directory (passed to `-t [dir]` in `cmsTriton` if non-empty; if `"."`, uses `cmsTriton` default)
* `imageName`: server image name (passed to `-i [name]` in `cmsTriton` if non-empty)
* `sandboxName`: Singularity sandbox directory (passed to `-s [dir]` in `cmsTriton` if non-empty)

## Examples

Several example producers (running image classification networks or Graph Attention Network) can be found in the [test](./test) directory.

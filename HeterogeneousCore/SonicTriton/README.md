# SONIC for Triton Inference Server

Triton Inference Server ([docs](https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton_inference_server_1130/user-guide/docs/index.html), [repo](https://github.com/NVIDIA/triton-inference-server))
is an open-source product from Nvidia that facilitates the use of GPUs as a service to process inference requests.

Triton supports multiple named inputs and outputs with different types. The allowed types are:
boolean, unsigned integer (8, 16, 32, or 64 bits), integer (8, 16, 32, or 64 bits), floating point (16, 32, or 64 bit), or string.

Triton additionally supports inputs and outputs with multiple dimensions, some of which might be variable (denoted by -1).
Concrete values for variable dimensions must be specified for each call (event).

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
* `batchSize`: number of objects sent per request
  * can also be set on per-event basis using `setBatchSize()`
  * some models don't support batching
* `address`: server IP address
* `port`: server port
* `timeout`: maximum allowed time for a request
* `outputs`: optional, specify which output(s) the server should send

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
* `TritonInput<T> = std::vector<std::vector<T>>`
* `TritonOutput<T> = std::vector<edm::Span<const T*>>`

In a SONIC Triton producer, the basic flow should follow this pattern:
1. `acquire()`:  
    a. access input object(s) from `TritonInputMap`  
    b. allocate input data using `std::make_shared<TritonInput<T>>()`  
    c. fill input data  
    d. set input shape(s) (optional, only if any variable dimensions)  
    e. convert using `toServer()` function of input object(s)  
2. `produce()`:  
    a. access output object(s) from `TritonOutputMap`  
    b. obtain output data as `TritonOutput<T>` using `fromServer()` function of output object(s) (sets output shape(s) if variable dimensions exist)  
    c. fill output products  

Several example producers (running ResNet50 or Graph Attention Network), along with instructions to run a local server,
can be found in the [test](./test) directory.

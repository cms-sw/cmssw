# SONIC core infrastructure

SONIC: Services for Optimized Network Inference on Coprocessors

## For analyzers

The `SonicEDProducer` class template extends the basic Stream producer module in CMSSW.
Similarly, `SonicEDFilter` extends the basic Stream filter module (replace `void produce` with `bool filter` below).

To implement a concrete derived producer class, the following skeleton can be used:
```cpp
#include "HeterogeneousCore/SonicCore/interface/SonicEDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"

class MyProducer : public SonicEDProducer<Client>
{
public:
  explicit MyProducer(edm::ParameterSet const& cfg) : SonicEDProducer<Client>(cfg) {
    //do any necessary operations
  }
  void acquire(edm::Event const& iEvent, edm::EventSetup const& iSetup, Input& iInput) override {
    //convert event data to client input format
  }
  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup, Output const& iOutput) override {
    //convert client output to event data format
  }
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
    edm::ParameterSetDescription desc;
    Client::fillPSetDescription(desc);
    //add producer-specific parameters
    descriptions.add("MyProducer",desc);
  }
};

DEFINE_FWK_MODULE(MyProducer);
```

The generic `Client` must be replaced with a concrete client (see next section), which has specific input and output types.

The python configuration for the producer should include a dedicated `PSet` for the client parameters:
```python
process.MyProducer = cms.EDProducer("MyProducer",
    Client = cms.PSet(
        # necessary client options go here
        mode = cms.string("Sync"),
        allowedTries = cms.untracked.uint32(0),
    )
)
```
These parameters can be prepopulated and validated by the client using `fillDescriptions()`.
The `mode` and `allowedTries` parameters are always necessary (example values are shown here, but other values are also allowed).
These parameters are described in the next section.

In addition, there is a `SonicOneEDAnalyzer` class template for user analysis, e.g. to produce simple ROOT files.
Only `Sync` mode is supported for clients used with One modules,
but otherwise, the above template can be followed (replace `void produce(edm::Event&` with `void analyze(edm::Event const&` above).

Examples of the producer, filter, and analyzer can be found in the [test](./test) folder.

## For developers

To add a new communication protocol for SONIC, follow these steps:
1. Submit the communication protocol software and any new dependencies to [cmsdist](https://github.com/cms-sw/cmsdist) as externals
2. Set up the concrete client(s) that use the communication protocol in a new package in the `HeterogeneousCore` subsystem
3. Add a test producer (see above) to make sure it works

To implement a concrete client, the following skeleton can be used for the `.h` file:
```cpp
#ifndef HeterogeneousCore_MyPackage_MyClient
#define HeterogeneousCore_MyPackage_MyClient

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HeterogeneousCore/SonicCore/interface/SonicClient.h"

class MyClient : public SonicClient<Input,Output> {
public:
  MyClient(const edm::ParameterSet& params, const std::string& debugName);

  static void fillPSetDescription(edm::ParameterSetDescription& iDesc);

protected:
  void evaluate() override;
};

#endif
```

The concrete client member function implementations, in an associated `.cc` file, should include the following:
```cpp
MyClient::MyClient(const edm::ParameterSet& params, const std::string& debugName)
    : SonicClient(params, debugName, "MyClient") {
  //do any necessary operations
}
```

The `SonicClient` has three available modes:
* `Sync`: synchronous call, blocks until the result is returned.
* `Async`: asynchronous, non-blocking call.
* `PseudoAsync`: turns a synchronous, blocking call into an asynchronous, non-blocking call, by waiting for the result in a separate `std::thread`.

`Async` is the most efficient, but can only be used if asynchronous, non-blocking calls are supported by the communication protocol in use.

In addition, as indicated, the input and output data types must be specified.
(If both types are the same, only the input type needs to be specified.)
The client constructor can optionally provide a value for `clientName_`,
which will be used in output messages alongside the debug name set by the producers.

In all cases, the implementation of `evaluate()` must call `finish()`.
For the `Sync` and `PseudoAsync` modes, `finish()` should be called at the end of `evaluate()`.
For the `Async` mode, `finish()` should be called inside the communication protocol callback function (implementations may vary).

When `finish()` is called, the success or failure of the call should be conveyed.
If a call fails, it can optionally be retried. This is only allowed if the call failure does not cause an exception.
Therefore, if retrying is desired, any exception should be converted to a `LogWarning` or `LogError` message by the client.
A Python configuration parameter can be provided to enable retries with a specified maximum number of allowed tries.

The client must also provide a static method `fillPSetDescription()` to populate its parameters in the `fillDescriptions()` for the producers that use the client:
```cpp
void MyClient::fillPSetDescription(edm::ParameterSetDescription& iDesc) {
  edm::ParameterSetDescription descClient;
  fillBasePSetDescription(descClient);
  //add parameters
  iDesc.add<edm::ParameterSetDescription>("Client",descClient);
}
```

As indicated, the `fillBasePSetDescription()` function should always be applied to the `descClient` object,
to ensure that it includes the necessary parameters.
(Calling `fillBasePSetDescription(descClient, false)` will omit the `allowedTries` parameter, disabling retries.)

Example client code can be found in the `interface` and `src` directories of the other Sonic packages in this repository.

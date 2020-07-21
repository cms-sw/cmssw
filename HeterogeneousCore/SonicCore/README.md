# SONIC core infrastructure

SONIC: Services for Optimized Network Inference on Coprocessors

## For analyzers

The `SonicEDProducer` class template extends the basic Stream producer module in CMSSW.

To implement a concrete derived producer class, the following skeleton can be used:
```cpp
#include "HeterogeneousCore/SonicCore/interface/SonicEDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"

class MyProducer : public SonicEDProducer<Client>
{
public:
  explicit MyProducer(edm::ParameterSet const& cfg) : SonicEDProducer<Client>(cfg) {
    //for debugging
    setDebugName("MyProducer");
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

An example producer can be found in the [test](./test) folder.

## For developers

To add a new communication protocol for SONIC, follow these steps:
1. Submit the communication protocol software and any new dependencies to [cmsdist](https://github.com/cms-sw/cmsdist) as externals
2. Set up the concrete client(s) that use the communication protocol in a new package in the `HeterogeneousCore` subsystem
3. Add a test producer (see above) to make sure it works

To implement a concrete client, the following skeleton can be used for the `.h` file, with the function implementations in an associated `.cc` file:
```cpp
#ifndef HeterogeneousCore_MyPackage_MyClient
#define HeterogeneousCore_MyPackage_MyClient

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HeterogeneousCore/SonicCore/interface/SonicClient.h"

class MyClient : public SonicClient<Input,Output> {
public:
  MyClient(const edm::ParameterSet& params);

  static void fillPSetDescription(edm::ParameterSetDescription& iDesc);

protected:
  void evaluate() override;
};

#endif
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
  edm::ParameterSetDescription descClient(basePSetDescription());
  //add parameters
  iDesc.add<edm::ParameterSetDescription>("Client",descClient);
}
```

As indicated, the `descClient` object should always be initialized using the `basePSetDescription()` function,
to ensure that it includes the necessary parameters.
(Calling `basePSetDescription(false)` will omit the `allowedTries` parameter, disabling retries.)

Example client code can be found in the `interface` and `src` directories of the other Sonic packages in this repository.

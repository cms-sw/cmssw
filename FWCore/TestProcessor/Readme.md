# FWCore/TestProcessor Documentation

## Introduction
The `TestProcessor` class is used to test individual CMSSW framework modules. The class allows one to 

* Specify a string containing the python configuration of the module and
* Pass data from the test to the module through the edm::Event and/or the edm::EventSetup

The system is composed of three parts

1. A python `TestProcess` class
1. A C++ configuration class
1. The C++ `TestProcessor` class

## Python `TestProcess` class
The `TestProcess` class has all the same attributes as the standard `cms.Process` class except

* If a process name is not give, it will default to `"TEST"`
* `cms.Path` and `cms.EndPath` are ignored
* the method `moduleToTest` was added and is used to specify which module is to be called during the test. This method also accepts a `cms.Task` which is used to run any other modules that may be needed.

As stated above, additional framework modules and EventSetup modules and Services are allowed to be specified in the configuration. If no EventSetup modules are specified in the `cms.Task` passed to `moduleToTest` than all EventSetup modules loaded into `TestProcess` will be created for the test. The same holds for Services.

### Example 1: Setup only the module

```python
from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.foo = cms.EDProducer('FooProd')
process.moduleToTest(process.foo)
```

### Example 2: Use additional modules

```python
from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.load("somePackage.someModules_cff")
process.foo = cms.EDProducer('FooProd')
process.moduleToTest(process.foo, cms.Task(process.bar))
```

NOTE: We recommend testing modules completely in isolation, however we realize, for some cases, that is not practical.

## C++ Configuration Class
The configuration class `edm::test::TestProcessor::Config` is used to 

* hold a C++ string containing the python configuration using `TestProcess`
* register Event data products that the test may be using
* register EventSetup data products that th etest may be using
* register additional process names which simulates reading data from a file

The configuration class is separate from the `edm::test::TestProcessor` class since once the latter is setup, one can not change its behavior (i.e. you can not add additional event data products to produce).

The same configuration class can be reused to setup multiple `edm::test::TestProcessor` instances.

### Example 1: Use only the python configuration
We recommend using a C++ raw string for the python configuration. That way line breaks and quotation marks are automatically handled.

```cpp
edm::test::TestProcessor::Config config{
R"_(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.foo = cms.EDProducer("FooProd")
process.moduleToTest(process.foo)
)_"
};
```
    
### Example 2: Adding Event data products
You must register which Event data products you intend to pass to the test. A `edm::EDPutTokenT<>` object is returned from the registration call. This object must be passed to the test along with an `std::unique_ptr<>` containing the object. Multiple Event data products are allowed.

```cpp
edm::test::TestProcessor::Config config{
R"_(...
)_"
};

//Uses the module label 'bar'
auto barPutToken = config.produces<std::vector<Bar>>("bar");
```
    
### Example 3: Adding Event data products from an earlier Process
To add an addition process to the test, one must call the `addExtraProcess` method. This method returns a `edm::test::ProcessToken`. This token is then passed to the `produces` call. If a `edm::test::ProcessToken` is not passed, the data product is set to come from the most _recent_ process.
One can specify multiple extra processes. The order of the calls determines the order of the process histories. With the first call creating the oldest process.

```cpp
auto hltProcess = config.addExtraProcess("HLT");
auto barPutToken = config.produces<std::vector<Bar>>("bar","",hltProcess);
```

###Example 4: Adding EventSetup data products
You must register which EventSetup data products you intend to pass to the test as well as the EventSetup Record from which the data product can be obtained. A `edm::test::ESPutTokenT<>` object is returned from the registration call. This object must be passed to the test along with an `std::unique_ptr<>` containing the object. Multiple EventSetup data products are allowed.

```cpp
auto esPutToken = config.esProduces<FooData,FooRecord>();
```

## `TestProcessor` class

`edm::test::TestProcessor` does the work of running the tests via its `test()` method. An instance of the class is constructed by passing it an instance of `edm::test::TestProcessor::Config`. In the constructor, the class will load all the framework modules needed by the python configuration as well as setup all internal details needed to run CMSSW framework transitions.

The first call to `test()` will cause the _beginJob_, _beginStream_, _globalBeginRun_, _streamBeginRun_, _globalBeginLuminosityBlock_ and _streamBeginLuminosityBlock_ transitions of the modules to be called in addition to the _event_ transition. Subsequent calls to `test()` will only generate new _event_ transitions. 

Calling `setRunNumber()` or `setLuminosityBlockNumber()` will trigger the appropriate _begin_ transition calls and will be preceeded by the appropriate _end_ transition calls if a call to `test()` was already made.

The `test()` method accepts an unlimited number of arguments of the type `std::pair<edm::EDPutTokenT<T>,std::unique_ptr<T>>` and `std::pair<edm::test::ESPutTokenT<T>, std::unique_ptr<T>`. Each call to `test()` will also generate a new EventSetup IOV for the Records declared during the EventSetup data product registration calls to `edm::test::TestProcessor::Config`. This allows each call to `test()` to update the EventSetup data product to be used.
  
### `edm::test::Event`
The return value of `test()` is an `edm::test::Event`. The `Event` class gives access to any data products created by the module being tested via a call to `get<T>()`. Since the module label and the process name are already known by the test, `get<T>()` takes one optional argument which is the _productInstanceLabel_ for the data product. The call to `get<T>()` returns an `edm::test::TestHandle<T>` which acts like a smart pointer to the data product if the product was retrieved. If the data product was not retrieved, attempting to access the data will cause an exception.
  
The `edm::test::Event` also has the method `modulePassed()` which is only useful when testing an `EDFilter`. In that case, the method returns `true` if the module passed the Event and `false` otherwise.

### Run and LuminosityBlock product testing
It is possible to also test Run and LuminosityBlock products created by the module. This can be accomplished by calling
* `testBeginRun(edm::RunNumber_t)`
* `testEndRun()`
* `testBeginLuminosityBlock(edm::LuminosityBlockNumber_t)`
* `testEndLuminosityBlock()`

Like `test()` one can pass an unlimited number of arguments of type  `std::pair<edm::test::ESPutTokenT<T>, std::unique_ptr<T>` in order to control EventSetup data passed to the module (`EDPutToken` are not supported at this time). Each of the above `test` functions also return either `edm::test::Run` or `edm::test::LuminosityBlock` which has the same interface as `edm::test::Event` except they do not have the method `modulePassed()` because filtering does not occur on those transitions. 

The `test` methods all make sure all the needed transitions occur in the necessary order. For example, calling `test()` and then `testEndRun()` will cause _streamEndLuminosityBlock_ and _globalEndLuminosityBlock_ to occur.

When calling `testBeginRun(edm:RunNumber_t)` or `testBeginLuminosityBlock(edm::LuminosityBlockNumber_t)` one is required to pass in a Run or LuminosityBlock number which is different from the number previously used by any earlier calls to a `test` function. 


### Full Example 

```cpp
#include "FWCore/TestProcessor/interface/TestProcessor.h"
int main() {
  //The python configuration
  edm::test::TestProcessor::Config config{
  R"_(from FWCore.TestProcessor.TestProcess import *
  process = TestProcess()
  process.foo = cms.EDProducer("FooProd")
  process.moduleToTest(process.foo)
  )_"
  };

  //setup data to pass
  auto barPutToken = config.produces<std::vector<Bar>>("bar");
  auto esPutToken = config.esProduces<FooData,FooRecord>();

  edm::test::TestProcessor tester{ config };

  //Run a test
  auto event = tester.test(std::make_pair(barPutToken, 
                                          std::make_unique<std::vector<Bar>(...)),
                           std::make_pair(esPutToken,
                                          std::make_unique<FooData>(...)));
  auto nMade = event.get<std::vector<Foo>>()->size();
  std::cout << nMade <<std::endl;
  
  if( nMade != ...) {
    return 1;
  }
  return 0;
};
```

### Example using Catch2

[Catch2](https://github.com/catchorg/Catch2/blob/master/docs/Readme.md#top) is a simple to use C++ unit testing framemwork. It can be used in conjuction with `TestProcessor` to drive a series of tests. In addition to the code, be sure to add `<use name="catch2"/>` to the `BuildFile.xml`.

```cpp
#include "FWCore/TestProcessor/interface/TestProcessor.h"
...
#include "catch.hpp"

TEST_CASE("FooProd tests", "[FooProd]") {
  //The python configuration
  edm::test::TestProcessor::Config config{
R"_(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.foo = cms.EDProducer("FooProd")
process.moduleToTest(process.foo)
)_"
  };

  //setup data to pass
  auto barPutToken = config.produces<std::vector<Bar>>("bar");
  auto esPutToken = config.esProduces<FooData,FooRecord>();

  SECTION("Pass standard data") {
    edm::test::TestProcessor tester{ config };

    //Run a test
    auto event = tester.test(std::make_pair(barPutToken, 
                                            std::make_unique<std::vector<Bar>(...)),
                             std::make_pair(esPutToken,
                                            std::make_unique<FooData>(...)));
    auto const& foos = event.get<std::vector<Foo>>();
    REQUIRE(foos->size() == ...);
    REQUIRE(foos[0] == Foo(...));
    ...
    
    SECTION("Move to new IOV") {
      tester.setRunNumber(2);
      auto event = tester.test(std::make_pair(barPutToken, 
                                              std::make_unique<std::vector<Bar>(...)),
                               std::make_pair(esPutToken,
                                              std::make_unique<FooData>(...)));
      auto const& foos = event.get<std::vector<Foo>>();
      REQUIRE(foos->size() == ...);
      REQUIRE(foos[0] == Foo(...));
      ...
    };
  };
  
  SECTION("Missing event data") {
    edm::test::TestProcessor tester{ config };
    REQUIRE_THROWS_AS(tester.test(std::make_pair(esPutToken,
                                              std::make_unique<FooData>(...))), 
                      cms::Exception);
  };
}
```

## Autogenerating Tests

Tests for new modules are automatically created when using `mkedprod`, `mkedfltr` or `mkedanlzr`. The same commands can be used to generate tests for existing modules just by running those commands from within the `test` directory of the package containing the module. For this case, you will need to manually add the following to `test/BuildFile.xml`:

```xml
<bin file="test_catch2_*.cc" name="test<SubSystem name><Package Name>TP">
<use name="FWCore/TestProcessor"/>
<use name="catch2"/>
</bin>
```


## Tips

### Testing different module configuration parameters

It will often be the case that a group of tests only differ based on the values of parameters used to configure a module. For this case, one would like to have a _base configuration_ which is used as the starting point for each of tests. Rather than using string manipulation to create each new configuration, we can make use of the fact we will be running a python interpreter on each configuration.

The _base configuration_ is a full configuration where, instead of setting specific values for each parameter that needs to be varied, we use a python variable name, where the variable is not declared in the _base configuration_.

```cpp
const std::string baseConfig = 
R"_(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.foo = cms.EDProducer('FooProd', value = cms.int32( fooValue ))
process.moduleToTest(process.foo)
)_";
```

To generate a specific configuration, we just prepend to the `baseConfig` a string containing an expressiong which sets the values on the variable names

```cpp
std::string fullConfig = "fooValue = 3\n"+baseConfig;
```
    
Alternatively, you can setup default values in the base configuration

```cpp
const std::string baseConfig = 
R"_(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.foo = cms.EDProducer('FooProd', value = cms.int32( 1 ))
process.moduleToTest(process.foo)
)_";
```

And then append a string at the end which sets the particular value

```cpp
std::string fullConfig = baseConfig +
R"_(process.foo.value = 3
)_"
```


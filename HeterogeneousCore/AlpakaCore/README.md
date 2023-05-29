# Alpaka algorithms and modules in CMSSW

## Introduction

This page documents the Alpaka integration within CMSSW. For more information about Alpaka itself see the [Alpaka documentation](https://alpaka.readthedocs.io/en/latest/).

### Compilation model

The code in `Package/SubPackage/{interface,src,plugins,test}/alpaka` is compiled once for each enabled Alpaka backend. The `ALPAKA_ACCELERATOR_NAMESPACE` macro is substituted with a concrete, backend-specific namespace name in order to guarantee different symbol names for all backends, that allows for `cmsRun` to dynamically load any set of the backend libraries.

The source files with `.dev.cc` suffix are compiled with the backend-specific device compiler. The other `.cc` source files are compiled with the host compiler.

The `BuildFile.xml` must contain `<flags ALPAKA_BACKENDS="1"/>` to enable the behavior described above.

## Overall guidelines

* Minimize explicit blocking synchronization calls
  * Avoid `alpaka::wait()`, non-cached memory buffer allocations
* If you can, use `global::EDProducer` base class
  * If you need per-stream storage
    * For few objects consider using [`edm::StreamCache<T>`](https://twiki.cern.ch/twiki/bin/view/CMSPublic/FWMultithreadedFrameworkGlobalModuleInterface#edm_StreamCacheT) with the global module, or
    * Use `stream::EDProducer`
  * If you need to transfer some data back to host, use `stream::SynchronizingEDProducer`
* All code using `ALPAKA_ACCELERATOR_NAMESPACE` should be placed in `Package/SubPackage/{interface,src,plugins,test}/alpaka` directory
  * Alpaka-dependent code that uses templates instead of the namespace macro can be placed in `Package/SubPackage/interface` directory
* All source files (not headers) using Alpaka device code (such as kernel call, functions called by kernels) must have a suffic `.dev.cc`, and be placed in the aforementioned `alpaka` subdirectory
* Any code that `#include`s a header from the framework or from the `HeterogeneousCore/AlpakaCore` must be separated from the Alpaka device code, and have the usual `.cc` suffix.
  * Some framework headers are allowed to be used in `.dev.cc` files:
    * Any header containing only macros, e.g. `FWCore/Utilities/interface/CMSUnrollLoop.h`, `FWCore/Utilities/interface/stringize.h`
    * `FWCore/Utilities/interface/Exception.h`
    * `FWCore/MessageLogger/interface/MessageLogger.h`, although it is preferred to issue messages only in the `.cc` files
    * `HeterogeneousCore/AlpakaCore/interface/EventCache.h` and `HeterogeneousCore/AlpakaCore/interface/QueueCache.h` can, in principle, be used in `.dev.cc` files, even if there should be little need to use them explicitly

## Data formats

Data formats, for both Event and EventSetup, should be placed following their usual rules. The Alpaka-specific conventions are
* There must be a host-only flavor of the data format that is either independent of Alpaka, or depends only on Alpaka's Serial backend
  * The host-only data format must be defined in `Package/SubPackage/interface/` directory
  * If the data format is to be serialized (with ROOT), it must be serialized in a way that the on-disk format does not depend on Alpaka, i.e. it can be read without Alpaka
  * For Event data products the ROOT dictionary should be defined in `DataFormats/SubPackage/src/classes{.h,_def.xml}`
    * As usual, the `classes_def.xml` should declare the dictionaries for the data product type `T` and `edm::Wrapper<T>`. These data products can be declared as persistent (default) or transient (`persistent="false"` attribute).
  * For EventSetup data products [the registration macro `TYPELOOKUP_DATA_REG`](https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideHowToRegisterESData) should be placed in `Package/SubPackage/src/ES_<type name>.cc`.
* The device-side data formats are defined in `Package/SubPackage/interface/alpaka/` directory
  * The device-side data format classes should be either templated over the device type, or defined in the `ALPAKA_ACCELERATOR_NAMESPACE` namespace.
  * For Event data products the ROOT dictionary should be defined in `DataFormats/SubPackage/src/alpaka/classes_<platform>{.h,_def.xml}`
    * The `classes_<platform>_def.xml` should declare the dictionaries for the data product type `T`, `edm::DeviceProduct<T>`, and `edm::Wrapper<edm::DeviceProduct<T>>`. All these dictionaries must be declared as transient with `persistent="false"` attribute.
    * The list of `<platform>` includes currently: `cuda`, `rocm`
  * For EventSetup data products the registration macro should be placed in `Package/SubPackage/src/alpaka/ES_<type name>.cc`
     * Data products defined in `ALPAKA_ACCELERATOR_NAMESPACE` should use `TYPELOOKUP_ALPAKA_DATA_REG` macro
     * Data products templated over the device type should use `TYPELOOKUP_ALPAKA_TEMPLATED_DATA_REG` macro
 * For Event data products the `DataFormats/SubPackage/BuildFile.xml` must contain `<flags ALPAKA_BACKENDS="cuda rocm"/>` (unless the package has something that is really specific for `serial` backend that is not generally applicable on host)

Note that even if for Event data formats the examples above used `DataFormats` package, Event data formats are allowed to be defined in other packages too in some circumstances. For full details please see [SWGuideCreatingNewProducts](https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideCreatingNewProducts).

### Implicit data transfers

Both EDProducers and ESProducers make use of implicit data transfers.

#### EDProducer

In EDProducers for each device-side data product a transfer from the device memory space to the host memory space is registered automatically. The data product is copied only if the job has another EDModule that consumes the host-side data product. The framework code to issue the transfer makes use of `cms::alpakatools::CopyToHost` class template that must be specialized along
```cpp
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"

namespace cms::alpakatools {
  template <>
  struct CopyToHost<TSrc> {
    template <typename TQueue>
    static auto copyAsync(TQueue& queue, TSrc const& deviceProduct) -> TDst {
      // code to construct TDst object, and launch the asynchronous memcpy from the device of TQueue to the host
      return ...;
    }
  };
}
```
Note that the destination (host-side) type `TDst` can be different from or the same as the source (device-side) type `TSrc` as far as the framework is concerned. For example, in the `PortableCollection` model the types are different. The `copyAsync()` member function is easiest to implement as a template over `TQueue`. The framework handles the necessary synchronization between the copy function and the consumer in a non-blocking way.

The `CopyToHost` class template is partially specialized for all `PortableCollection` instantiations.

#### ESProducer

In ESProducers for each host-side data product a transfer from the host memory space to the device memory space (of the backend of the ESProducer) is registered automatically. The data product is copied only if the job has another ESProducer or EDModule that consumes the device-side data product. The framework code to issue makes use of `cms::alpakatools::CopyToDevice` class template that must be specialized along 
```cpp
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToDevice.h"

namespace cms::alpakatools {
  template<>
  struct CopyToDevice<TSrc> {
    template <typename TQueue>
    static auto copyAsync(TQueue& queue, TSrc const& hostProduct) -> TDst {
      // code to construct TDst object, and launch the asynchronous memcpy from the host to the device of TQueue
      return ...;
    }
  };
}
```
Note that the destination (device-side) type `TDst` can be different from or the same as the source (host-side) type `TSrc` as far as the framework is concerned. For example, in the `PortableCollection` model the types are different. The `copyAsync()` member function is easiest to implement as a template over `TQueue`. The framework handles the necessary synchronization between the copy function and the consumer (currently the synchronization blocks, but work is ongoing to make it non-blocking).

The `CopyToDevice` class template is partially specialized for all `PortableCollection` instantiations.

### `PortableCollection`

For more information see [`DataFormats/Portable/README.md`](../../DataFormats/Portable/README.md) and [`DataFormats/SoATemplate/README.md`](../../DataFormats/SoATemplate/README.md).


## Modules

### Base classes

The Alpaka-based EDModules should use one of the following base classes (that are defined in the `ALPAKA_ACCELERATOR_NAMESPACE`):

* `global::EDProducer<...>` (`#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"`)
   * A [global EDProducer](https://twiki.cern.ch/twiki/bin/view/CMSPublic/FWMultithreadedFrameworkGlobalModuleInterface) that launches (possibly) asynchronous work
* `stream::EDProducer<...>` (`#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"`)
   * A [stream EDProducer](https://twiki.cern.ch/twiki/bin/view/CMSPublic/FWMultithreadedFrameworkStreamModuleInterface) that launches (possibly) asynchronous work
* `stream::SynchronizingEDProducer<...>` (`#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"`)
   * A [stream EDProducer](https://twiki.cern.ch/twiki/bin/view/CMSPublic/FWMultithreadedFrameworkStreamModuleInterface) that may launch (possibly) asynchronous work, and synchronizes the asynchronous work on the device with the host
      * The base class uses the [`edm::ExternalWork`](https://twiki.cern.ch/twiki/bin/view/CMSPublic/FWMultithreadedFrameworkStreamModuleInterface#edm_ExternalWork) for the non-blocking synchronization

The `...` can in principle be any of the module abilities listed in the linked TWiki pages, except the `edm::ExternalWork`. The majority of the Alpaka EDProducers should be `global::EDProducer` or `stream::EDProducer`, with `stream::SynchronizingEDProducer` used only in cases where some data to be copied from the device to the host, that requires synchronization, for different reason than copying an Event data product from the device to the host.

New base classes (or other functionality) can be added based on new use cases that come up.

The Alpaka-based ESProducers should use the `ESProducer` base class (`#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"`). 


### Event, EventSetup, Records

The Alpaka-based modules have a notion of a _host memory space_ and _device memory space_ for the Event and EventSetup data products. The data products in the host memory space are accessible for non-Alpaka modules, whereas the data products in device memory space are available only for modules of the specific Alpaka backend. The host backend(s) use the host memory space directly.

The EDModules get `device::Event` and `device::EventSetup` from the framework, from which data products in both host memory space and device memory space can be accessed. Data products can also be produced to either memory space. For all data products produced in the device memory space an implicit data copy from the device memory space to the host memory space is registered as discussed above. The `device::Event::queue()` returns the Alpaka `Queue` object into which all work in the EDModule must be enqueued. 

The ESProducer can have two different `produce()` function signatures
* If the function has the usual `TRecord const&` parameter, the function can read an ESProduct from the host memory space, and produce another product into the host memory space. An implicit copy of the data product from the host memory space to the device memory space (of the backend of the ESProducer) is registered as discussed above.
* If the function has `device::Record<TRecord> const&` parameter, the function can read an ESProduct from the device memory space, and produce another product into the device memory space. No further copies are made by the framework. The `device::Record<TRecord>::queue()` gives the Alpaka `Queue` object into which all work in the ESProducer must be enqueued. 

### Tokens

The memory spaces of the consumed and (in EDProducer case) produced data products are driven by the tokens. The token types to be used in different cases are summarized below. 


|                                                                | Host memory space             | Device memory space              |
|----------------------------------------------------------------|-------------------------------|----------------------------------|
| Access Event data product of type `T`                          | `edm::EDGetTokenT<T>`         | `device::EDGetToken<T>`          |
| Produce Event data product of type `T`                         | `edm::EDPutTokenT<T>`         | `device::EDPutToken<T>`          |
| Access EventSetup data product of type `T` in Record `TRecord` | `edm::ESGetToken<T, TRecord>` | `device::ESGetToken<T, TRecord>` |

With the device memory space tokens the type-deducing `consumes()`, `produces()`, and `esConsumes()` calls must be used (i.e. do not specify the data product type as part of the function call). For more information on these registration functions see
* [`consumes()` in EDModules](https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideEDMGetDataFromEvent#consumes)
* [`produces()` in EDModules](https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideCreatingNewProducts#Producing_the_EDProduct)
* [`esConsumes()` in EDModules](https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideHowToGetDataFromES#In_ED_module)
* [`consumes()` in ESProducers](https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideHowToGetDataFromES#In_ESProducer)


### `fillDescriptions()`

In the [`fillDescriptions()`](https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideConfigurationValidationAndHelp) function specifying the module label automatically with the [`edm::ConfigurationDescriptions::addWithDefaultLabel()`](https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideConfigurationValidationAndHelp#Automatic_module_labels_from_plu) is strongly recommended. Currently a `cfi` file is generated for a module for each Alpaka backend such that the backend namespace is explicitly used in the module definition. An additional `cfi` file is generated for the ["module type resolver"](#module-type-resolver-portable) functionality, where the module type has `@alpaka` postfix.

Also note that the `fillDescription()` function must have the same content for all backends, i.e. any backend-specific behavior with e.g. `#ifdef` or `if constexpr` are forbidden.

## Guarantees

* All Event data products in the device memory space are guaranteed to be accessible only for operations enqueued in the `Queue` given by `device::Event::queue()` when accessed through the `device::Event`.
* All Event data products in the host memory space are guaranteed to be accessible for all operations (after the data product has been obtained from the `edm::Event` or `device::Event`).
* All EventSetup data products in the device memory space are guaranteed to be accessible only for operations enqueued in the `Queue` given by `device::Event::queue()` when accessed via the `device::EventSetup` (ED modules), or by `device::Record<TRecord>::queue()` when accessed via the `device::Record<TRecord>` (ESProducers).
* The EDM Stream does not proceed to the next Event until after all asynchronous work of the current Event has finished.
  * **Note**: currently this guarantee does not hold if the job has any EDModule that launches asynchronous work but does not explicitly synchronize or produce any device-side data products.

## Examples

For concrete examples see code in [`HeterogeneousCore/AlpakaTest`](../../HeterogeneousCore/AlpakaTest) and [`DataFormats/PortableTestObjects`](../../DataFormats/PortableTestObjects).

### EDProducer

This example shows a mixture of behavior from test code in [`HeterogeneousCore/AlpakaTest/plugins/alpaka/`](HeterogeneousCore/AlpakaTest/plugins/alpaka/)
```cpp
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
// + usual #includes for the used framework components, data format(s), record(s)

// Module must be defined in ALPAKA_ACCELERATOR_NAMESPACE
namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // Base class is defined in ALPAKA_ACCELEATOR_NAMESPACE as well (note, no edm:: prefix!)
  class ExampleAlpakaProducer : public global::EDProducer<> {
  public:
    ExampleAlpakaProducer(edm::ParameterSet const& iConfig)
        // produces() must not specify the product type, it is deduced from deviceToken_
        : deviceToken_{produces()}, size_{iConfig.getParameter<int32_t>("size")} {}

    // device::Event and device::EventSetup are defined in ALPAKA_ACCELERATOR_NAMESPACE as well
    void produce(edm::StreamID sid, device::Event& iEvent, device::EventSetup const& iSetup) const override {
      // get input data products
      auto const& hostInput = iEvent.get(getTokenHost_);
      auto const& deviceInput = iEvent.get(getTokenDevice_);
      auto const& deviceESData = iSetup.getData(esGetTokenDevice_);
    
      // run the algorithm, potentially asynchronously
      portabletest::TestDeviceCollection deviceProduct{size_, event.queue()};
      algo_.fill(event.queue(), hostInput, deviceInput, deviceESData, deviceProduct);

      // put the asynchronous product into the event without waiting
      // must use EDPutToken with emplace() or put()
      //
      // for a product produced with device::EDPutToken<T> the base class registers
      // a separately scheduled transformation function for the copy to host
      // the transformation function calls
      // cms::alpakatools::CopyToDevice<portabletest::TestDeviceCollection>::copyAsync(Queue&, portabletest::TestDeviceCollection const&)
      // function
      event.emplace(deviceToken_, std::move(deviceProduct));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      // All backends must have exactly the same fillDescriptions() content!
      edm::ParameterSetDescription desc;
      desc.add<int32_t>("size");
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    // use edm::EGetTokenT<T> to read from host memory space
    edm::EDGetTokenT<FooProduct> const getTokenHost_;
    
    // use device::EDGetToken<T> to read from device memory space
    device::EDGetToken<BarProduct> const getTokenDevice_;

    // use device::ESGetToken<T, TRecord> to read from device memory space
    device::ESGetToken<TestProduct, TestRecord> const esGetTokenDevice_;

    // use device::EDPutToken<T> to place the data product in the device memory space
    device::EDPutToken<portabletest::TestDeviceCollection> const deviceToken_;
    int32_t const size_;

    // implementation of the algorithm
    TestAlgo algo_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TestAlpakaProducer);

```

### ESProducer to reformat an existing ESProduct for use in device

```cpp
// Module must be defined in ALPAKA_ACCELERATOR_NAMESPACE
namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // Base class is defined in ALPAKA_ACCELEATOR_NAMESPACE as well (note, no edm:: prefix!)
  class ExampleAlpakaESProducer : public ESProducer {
  public:
    ExampleAlpakaESProducer(edm::ParameterSet const& iConfig) {
      // register the production function
      auto cc = setWhatProduced(this);
      // register consumed ESProduct(s)
      token_ = cc.consumes();
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      // All backends must have exactly the same fillDescriptions() content!
      edm::ParameterSetDescription desc;
      descriptions.addWithDefaultLabel(desc);
    }

    // return type can be
    // - std::optional<T> (T is cheap to move),
    // - std::unique_ptr<T> (T is not cheap to move),
    // - std::shared_ptr<T> (allows sharing between IOVs)
    //
    // the base class registers a separately scheduled function to copy the product on device memory
    // the function calls
    // cms::alpakatools::CopyToDevice<SimpleProduct>::copyAsync(Queue&, SimpleProduct const&)
    // function
    std::optional<SimpleProduct> produce(TestRecord const& iRecord) {
      // get input data
      auto const& hostInput = iRecord.get(token_);

      // allocate data product on the host memory
      SimpleProduct hostProduct;

      // fill the hostProduct from hostInput

      return std::move(hostProduct);
    }

  private:
    edm::ESGetToken<TestProduct, TestRecord> token_;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(ExampleAlpakaESProducer);
```

### ESProducer to derive a new ESProduct from an existing device-side ESProduct

```cpp
// Module must be defined in ALPAKA_ACCELERATOR_NAMESPACE
namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // Base class is defined in ALPAKA_ACCELEATOR_NAMESPACE as well (note, no edm:: prefix!)
  class ExampleAlpakaDeriveESProducer : public ESProducer {
  public:
    ExampleAlpakaDeriveESProducer(edm::ParameterSet const& iConfig) {
      // register the production function
      auto cc = setWhatProduced(this);
      // register consumed ESProduct(s)
      token_ = cc.consumes();
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      // All backends must have exactly the same fillDescriptions() content!
      edm::ParameterSetDescription desc;
      descriptions.addWithDefaultLabel(desc);
    }

    std::optional<OtherProduct> produce(device::Record<TestRecord> const& iRecord) {
      // get input data in the device memory space
      auto const& deviceInput = iRecord.get(token_);

      // allocate data product on the device memory
      OtherProduct deviceProduct(iRecord.queue());

      // run the algorithm, potentially asynchronously
      algo_.fill(iRecord.queue(), deviceInput, deviceProduct);

      // return the product without waiting
      return std::move(deviceProduct);
    }

  private:
    edm::ESGetToken<SimpleProduct, TestRecord> token_;
    
    OtherAlgo algo_;
  };

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(ExampleAlpakaDeviceESProducer);
```

## Configuration

There are a few different options for using Alpaka-based modules in the CMSSW configuration.

In all cases the configuration must load the necessary `ProcessAccelerator` objects (see below) For accelerators used in production, these are aggregated in `Configuration.StandardSequences.Accelerators_cff`. The `runTheMatrix.py` handles the loading of this `Accelerators_cff` automatically. The HLT menus also load the necessary `ProcessAccelerator`s.
```python
## Load explicitly
# One ProcessAccelerator for each accelerator technology
process.load("HeterogeneousCore.CUDACore.ProcessAcceleratorCUDA_cfi")

# And one ProcessAccelerator for Alpaka
process.load("HeterogeneousCore.AlpakaCore.ProcessAcceleratorAlpaka_cfi")

## Or, just load
process.load("Configuration.StandardSequences.Accelerators_cff")
```

### Explicit module type (non-portable)

The Alpaka modules can be used in the python configuration with their explicit, full type names
```python
process.producerCPU = cms.EDProducer("alpaka_serial_sync::ExampleAlpakaProducer", ...)
process.producerGPU = cms.EDProducer("alpaka_cuda_async::ExampleAlpakaProducer", ...)
```
Obviously this kind of configuration can be run only on machines that provide the necessary hardware. The configuration is thus explicitly non-portable.


### SwitchProducerCUDA (semi-portable)

A step towards a portable configuration is to use the `SwitchProcucer` mechanism, for which currently the only concrete implementation is [`SwitchProducerCUDA`](../../HeterogeneousCore/CUDACore/README.md#automatic-switching-between-cpu-and-gpu-modules). The modules for different Alpaka backends still need to be specified explicitly
```python
from HeterogeneousCore.CUDACore.SwitchProducerCUDA import SwitchProducerCUDA
process.producer = SwitchProducerCUDA(
    cpu = cms.EDProducer("alpaka_serial_sync::ExampleAlpakaProducer", ...),
    cuda = cms.EDProducer("alpaka_cuda_async::ExampleAlpakaProducer", ...)
)

# or

process.producer = SwitchProducerCUDA(
    cpu = cms.EDAlias(producerCPU = cms.EDAlias.allProducts(),
    cuda = cms.EDAlias(producerGPU = cms.EDAlias.allProducts()
)
```
This kind of configuration can be run on any machine (a given CMSSW build supports), but is limited to CMSSW builds where the modules for all the Alpaka backends declared in the configuration can be built (`alpaka_serial_sync` and `alpaka_cuda_async` in this example). Therefore the `SwitchProducer` approach is here called "semi-portable".

### Module type resolver (portable)

A fully portable way to express a configuration can be achieved with "module type resolver" approach. The module is specified in the configuration without the backend-specific namespace, and with `@alpaka` postfix
```python
process.producer = cms.EDProducer("ExampleAlpakaProducer@alpaka", ...)

# backend can also be set explicitly
process.producerCPU = cms.EDProducer("ExampleAlpakaProducer@alpaka",
    ...
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string("serial_sync")
    )
)
```
The `@alpaka` postfix in the module type tells the system the module's exact class type should be resolved at run time. The type (or backend) is set according to the value of `process.options.accelerators` and the set of accelerators available in the machine. If the backend is set explicitly in the module's `alpaka` PSet, the module of that backend will be used.

This approach is portable also across CMSSW builds that support different sets of accelerators, as long as only the host backends (if any) are specified explicitly in the `alpaka` PSet.

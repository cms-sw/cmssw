## HeterogeneousCore/AlpakaInterface

This package only depends on the `alpaka` header-only external library, and
provides the interface used by other packages in CMSSW.

It is safe to be used inside DataFormats packages.


## Memory operations

Alpaka buffers represent a multidimensional array in host or device memory.
Buffers implement a shared ownership model of the the memory they represent,
similar to an `std::shared_ptr<std::vector<T>>`.

Alpaka views represent a multidimensional array in host or device memory.
Views _do not_ own the memory, they refer to some pre-existing, underlying
object; as with raw pointers, the user must guarantee that the memory remains
valid for as long as the view is used.

This package defines in the `cms::alpakatools` namespace type aliases and helper
functions to simplify the use of zero-dimensional (scalars) and one-dimensional
(array) alpaka buffers and views, residing in host and device memory.


## Host memory buffers

These type aliases simplify the use of zero-dimensional (scalars) and
one-dimensional (array) alpaka host buffers:

  - `host_buffer<T>` (where `T` is not an array)
    represents a single object of type `T`;

  - `host_buffer<T[N]>`
    represents a one-dimensional buffer of `N` objects of type `T`;

  - `host_buffer<T[]>`
    represents a one-dimensional buffer of a number of objects of type `T`
    determined at construction time.


### `const` host memory buffers

As for an `std::shared_ptr`, the use of a `const` buffer is not particularly
usefeful, because one can always make a non-`const` copy that shares ownership
of the same underlying memory and objects.
When `const`-only access to a buffer is desired, an `alpaka::ViewConst` adapter
can be used to provide the required semantics.

These type aliases implement host buffers with `const`-only semantics:

  - `const_host_buffer<T>`;
  - `const_host_buffer<T[N]>`;
  - `const_host_buffer<T[]>`.

A `const_host_buffer` can be automatically constructued from a `host_buffer`.
These are mostly useful as function arguments, to guarantee that the function
does not make changes to the underlying memory.


### Allocating host buffers

These helper functions allocate zero-dimensional (scalars) and one-dimensional
(array) alpaka buffers in general purpose host memory:

  - `auto make_host_buffer<T>()` (where `T` is not an array)
    allocates a zero-dimensional buffer holding a single object of type `T`;

  - `auto make_host_buffer<T[N]>()`
    allocates a one-dimensional buffer holding `N` objects of type `T`;

  - `auto make_host_buffer<T[]>(size)`
    allocates a one-dimensional buffer holding `size` objects of type `T`.

These function allocate the buffers in general purpose host memory, using
immediate operations, without any caching.

The memory is not initialised.


### Allocating host buffers in device-mapped memory

These helper functions allocate zero-dimensional (scalars) and one-dimensional
(array) alpaka buffers in device-mapped memory:

  - `auto make_host_buffer<T, Platform>()` (where `T` is not an array)
    allocates a zero-dimensional buffer holding a single object of type `T`;

  - `auto make_host_buffer<T[N], Platform>()`
    allocates a one-dimensional buffer holding `N` objects of type `T`;

  - `auto make_host_buffer<T[], Platform>(size)`
    allocates a one-dimensional buffer holding `size` objects of type `T`.

These functions allocate the buffers in host memory that is pinned and mapped in
the address space of the devices, using potentially blocking operations, without
any caching. The memory and objects can be accessed both on the host and on the
devices.

The memory is not initialised.

On most systems accessing this memory from the device is significantly slower
than using device global memory, so it is not adviced to use these buffers
directly in device kernels. A common use case for device-mapped memory is as a
staging area for copies to and from the devices.


### Allocating queue-ordered host buffers in device-mapped memory

These helper functions allocate zero-dimensional (scalars) and one-dimensional
(array) alpaka queue-ordered buffers in device-mapped memory:

  - `auto make_host_buffer<T>(queue)` (where `T` is not an array)
    allocates a zero-dimensional buffer holding a single object of type `T`;

  - `auto make_host_buffer<T[N]>(queue)`
    allocates a one-dimensional buffer holding `N` objects of type `T`;

  - `auto make_host_buffer<T[]>(queue, size)`
    allocates a one-dimensional buffer holding `size` objects of type `T`.

These functions use queue-ordered, potentially cached, memory allocations.
Queue-ordered host buffers are available immediately. However, when they go out
of scope or are destroyed, the buffer memory and the underlying objects are kept
alive and remain valid until all the operations that have been enqueued in the
given queue up to that point have completed.

These functions allocate the buffers in host memory that is pinned and mapped in
the address space of the device associated to the `queue`; the memory and objects
can be accessed both on the host and on the devices from the same platform as the
device associated to the `queue`.

These function potentially use caching to reduce the allocation overhead.

The memory is not initialised.

See the previous section for considerations about the use of device-mapped
memory.


## Notes about copies and synchronisation

### Host-to-device copy

When copying data from a host buffer to a device buffer, _e.g._ with
```c++
alpaka::memcpy(queue, device_buffer, host_buffer);
```
the copy will run asynchronously along the given `queue`. This means that the
data can be accessed on the _device_ by any subsequent operation scheduled on
the same `queue`, like the execution of a kernel, a `memset`, a device-to-device
copy, _etc_. For example, this:
```c++
alpaka::memcpy(queue, device_buffer, host_buffer);
alpaka::exec(queue, workdivision, kernel{}, device_buffer.data());
```
is a safe approach that is commonly used.

However, the _host_ must avoid freeing, modifying or otherwise reusing the
buffer from which the data is being copied until the operation is complete.
For example, something like
```c++
alpaka::memcpy(queue, a_device_buffer, a_host_buffer);
std::memset(a_host_buffer.data(), 0x00, size);
```
is likely to overwrite part of the buffer while the copy is still ongoing,
resulting in `a_device_buffer` with incomplete and corrupted contents.

### Host-to-device move

For host data types that are movable and not copyable one can, to
large degree, avoid worrying about the caveats above about avoiding
any operations on the host with the following utility and move semantics
```c++
#include "HeterogeneousCore/AlpakaInterface/interface/moveToDeviceAsync.h"
// ...
auto device_object = cms::alpakatools::moveToDeviceAsync(queue, std::move(host_object));
```

Here the host-side `host_object` is _moved_ to the
`moveToDeviceAsync()` function, which returns a correponding
device-side `device_object`. In this case any subsequent use of
`host_object` is clearly "use after move", which is easier to catch in
code review or by static analysis tools than the consequences of
`alpaka::mempcy()`.

The `cms::alpakatools::CopyToDevice<T>` class temlate must have a
specialization for the host data type (otherwise the compilation will fail).

As mentioned above, the host data type must be movable but not
copyable (the compilation will fail with copyable types). For example,
the `PortableHostCollection` and `PortableHostObject` class templates
can be used, but Alpaka buffers can not be directly used.

The host data object should manage memory in
[queue-ordered](#allocating-queue-ordered-host-buffers-in-device-mapped-memory)
way. If not, the object must synchronize the device and the host in
its destructor (although such synchronization is undesirable).
Otherwise, the behavior is undefined.

### Device-to-host copy

When copying data from a device buffer to a host buffer, _e.g._ with
```c++
alpaka::memcpy(queue, a_host_buffer, a_device_buffer);
```
the copy will run asynchronously along the given `queue`. This means that the
data can be accessed on the _host_ by any subsequent operation scheduled on
the same `queue`, or after an explicit synchronisation.
A simple approach would be to wait for the copy to be complete:
```c++
alpaka::memcpy(queue, a_host_buffer, a_device_buffer);
alpaka::wait(queue);
int a_value = *a_host_buffer;
```
This has the downside of blocking the host until the copy - and all previously
enqueued operations - have completed. A more efficient approach is to use an
`edm::stream::EDProducer<edm::ExternalWork>` or a `stream::SynchronizingEDProducer`,
with the copy in the `acquire()` method, and the access to the data in the
`produce()` method:
```c++
// data member
host_buffer<int> a_host_buffer_;

// acquire() runs immediately and submits asynchronous operations to the device
void acquire(device::Event const& event, device::EventSetup const& setup) {
  ...
  alpaka::memcpy(event.queue(), a_host_buffer_, a_device_buffer);
}

// produce() is scheduled only once all operation in queue have completed
void produce(device::Event& event, device::EventSetup const& setup) {
  int a_value = *host_buffer_;
  ...
}
```


## Device buffers

These type aliases simplify the use of zero-dimensional (scalars) and
one-dimensional (array) alpaka device buffers:

  - `device_buffer<Device, T>` (where `T` is not an array)
    represents a single object of type `T`;

  - `device_buffer<Device, T[N]>`
    represents a one-dimensional buffer of `N` objects of type `T`;

  - `device_buffer<Device, T[]>`
    represents a one-dimensional buffer of a number of objects of type `T`
    determined at construction time.


### `const` device memory buffers

As for an `std::shared_ptr`, the use of a `const` buffer is not particularly
usefeful, because one can always make a non-`const` copy that shares ownership
of the same underlying memory and objects.
When `const`-only access to a buffer is desired, an `alpaka::ViewConst` adapter
can be used to provide the required semantics.

These type aliases implement device buffers with `const`-only semantics:

  - `const_device_buffer<Device, T>`;
  - `const_device_buffer<Device, T[N]>`;
  - `const_device_buffer<Device, T[]>`.

A `const_device_buffer` can be automatically constructued from a `device_buffer`.
These are mostly useful as function arguments, to guarantee that the function
does not make changes to the underlying memory.


### Allocating device buffers

These helper functions allocate zero-dimensional (scalars) and one-dimensional
(array) alpaka buffers in device global memory:

  - `auto make_device_buffer<T>(device)` (where `T` is not an array)
    allocates a zero-dimensional buffer holding a single object of type `T` in
    the global memory of `device`;

  - `auto make_device_buffer<T[N]>(device)`
    allocates a one-dimensional buffer holding `N` objects of type `T` in the
    global memory of `device`;

  - `auto make_device_buffer<T[]>(device, size)`
    allocates a one-dimensional buffer holding `size` objects of type `T` in the
    global memory of device.

These function allocate the buffers in device global memory, using potentially
blocking operations, without any caching.

The memory is not initialised.


### Allocating queue-ordered device buffers

These helper functions allocate zero-dimensional (scalars) and one-dimensional
(array) alpaka queue-ordered buffers in device global memory:

  - `auto make_device_buffer<T>(queue)` (where `T` is not an array)
    allocates a zero-dimensional buffer holding a single object of type `T` in
    the global memory of `device`;

  - `auto make_device_buffer<T[N]>(queue)`
    allocates a one-dimensional buffer holding `N` objects of type `T` in the
    global memory of `device`;

  - `auto make_device_buffer<T[]>(queue, size)`
    allocates a one-dimensional buffer holding `size` objects of type `T` in the
    global memory of device.

These functions use queue-ordered memory allocations.
Queue-ordered device buffers are available once all the operations that have
been enqueued in the given queue up to that point have completed.
In a similar manner, when the buffers go out of scope or are destroyed, the
buffer memory and the underlying objects are kept alive and remain valid until
all the operations that have been enqueued in the given queue up to that point
have completed.

These function allocate the buffers in device global memory, potentially using
caching to reduce the allocation overhead.

The memory is not initialised.


## Host memory views

These type aliases simplify the use of zero-dimensional (scalars) and
one-dimensional (array) alpaka host views:

  - `host_view<T>` (where `T` is not an array)
    represents a zero-dimensional view over a single object of type `T` in host
    memory;

  - `host_view<T[N]>`
    represents a one-dimensional view over `N` objects of type `T` in host
    memory;

  - `host_view<T[]>`
    represents a one-dimensional view over a number of objects of type `T` in
    host memory determined at construction time.


## Instantiatig host views

These helper functions instantiate zero-dimensional (scalars) and one-dimensional
(array) alpaka views over existing objects in host memory:

  - `auto make_host_view<T>(T& data)` (where `T` is not an array)
    instantiates a zero-dimensional view over the object `data` in host memory;

  - `auto make_host_view<T[N]>(T[N]& data)`
    instantiates a one-dimensional view over an array `data` in host memory;

  - `auto make_host_view<T[]>(T* data, size)`
    instantiates a one-dimensional view over an array starting at `data` with
    `size` elements in host memory;


## Device memory views

These type aliases simplify the use of zero-dimensional (scalars) and
one-dimensional (array) alpaka device views:

  - `device_view<Device, T>` (where `T` is not an array)
    represents a zero-dimensional view over a single object of type `T` in
    device global memory;

  - `device_view<Device, T[N]>`
    represents a one-dimensional view over `N` objects of type `T` in device
    global memory;

  - `device_view<Device, T[]>`
    represents a one-dimensional view over a number of objects of type `T` in
    device global memory determined at construction time.


## Instantiatig device views

These helper functions instantiate zero-dimensional (scalars) and one-dimensional
(array) alpaka views over existing objects in device memory:

  - `auto make_device_view<T>(device, T& data)` (where `T` is not an array)
    instantiates a zero-dimensional view over the object `data` in device global
    memory;

  - `auto make_device_view<T[N]>(device, T[N]& data)`
    instantiates a one-dimensional view over an array `data` in device global
    memory;

  - `auto make_device_view<T[]>(device, T* data, size)`
    instantiates a one-dimensional view over an array starting at `data` with
    `size` elements in device global memory.

The `make_device_view` functions can also accept as a first argument a `queue`
instead of a `device`:

  - `auto make_device_view<T>(queue, T& data)` (where `T` is not an array)
    instantiates a zero-dimensional view over the object `data` in device global
    memory;

  - `auto make_device_view<T[N]>(queue, T[N]& data)`
    instantiates a one-dimensional view over an array `data` in device global
    memory;

  - `auto make_device_view<T[]>(queue, T* data, size)`
    instantiates a one-dimensional view over an array starting at `data` with
    `size` elements in device global memory.

These functions use the device associated to the given `queue`. These operations
are otherwise identical to those that take a `device` as their first argument.

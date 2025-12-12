# `ngt::AnyBuffer`

`ngt::AnyBuffer` behaves like `std::any`, with two differences: it can only be
used with trivially copyable types, and provides access to the underlying memory
buffer to allow `memcpy`'ing its content.


# `ngt::MemoryCopyTraits<T>`

`struct ngt::MemoryCopyTraits<T>` should be specialised for each type that can
be safely `memcpy`'ed.

The specialisation shall have two static methods

    static std::vector<std::span<std::byte>> regions(T& object);
    static std::vector<std::span<const std::byte>> regions(T const& object);

that return a `vector` of `span` describing address, size pairs. A type that
supports this interface can be copied by doing a `memcpy` of all the memory
areas from a source object to a destination object.

A specialisation may implement the `initialize()` method, to initialize a newly
allocated object:

    static void initialize(T& object);


A specialisation may implement the method `properties()`, which should return
the properties (_e.g._ number of elements, or size in bytes, _etc_.) of an
existing object:

    using Properties = ...;
    static Properties properties(T const& object);

If `properties()` is defined, the value it returns should be passed as the
second argument to `initialize()`:

    static void initialize(T& object, Properties const& args););


A specialisation may provide the method `finalize()`:

    static void finalize(T& object);

If present, it should be called to restore the object invariants after a
`memcpy` operation.


## Specialisations of `ngt::MemoryCopyTraits`

This package provides the specialisation of `ngt::MemoryCopyTraits` for all
arithmetic types, all `std::vector`s of arithmetic types (except
`std::vector<bool>`), and `std::string`.


# `ngt::GenericCloner`

This `EDProducer` will clone all the event products declared in its
configuration, using either the plugin-based NGT trivial serialisation, or the
products' ROOT dictionaries.

The products can be specified either as module labels (_e.g._ `"<module label>"`)
or as branch names (_e.g._ `"<product type>_<module label>_<instance name>_<process name>"`).

If a module label is used, no underscore (`_`) must be present; this module will
clone all the products produced by that module, including those produced by the
Transformer functionality (such as the implicitly copied-to-host products in
case of Alpaka-based modules).
If a branch name is used, all four fields must be present, separated by
underscores; this module will clone only on the matching product(s).

Glob expressions (`?` and `*`) are supported in module labels and within the
individual fields of branch names, similar to an `OutputModule`'s "keep"
statements.
Use `"*"` to clone all products.

For example, in the case of Alpaka-based modules running on a device, using

    eventProducts = cms.untracked.vstring("module")

will cause "module" to run, along with automatic copy of its device products to
the host, and will attempt to clone all device and host products.
To clone only the host product, the branch can be specified explicitly with

    eventProducts = cms.untracked.vstring( "HostProductType_module_*_*" )

.

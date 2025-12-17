# `ngt::MemoryCopyTraits<T>`

`struct ngt::MemoryCopyTraits<T>` should be specialised for each type that can
be safely `memcpy`ed.

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

# `ngt::AnyBuffer`

`ngt::AnyBuffer` behaves like `std::any`, with two differences: it can only be
used with trivially copyable types, and provides access to the underlying memory
buffer to allow `memcpy`ing its content.


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
individual fields of branch names, similar to an `OutputModule`s "keep"
statements.
Use `"*"` to clone all products.

For example, in the case of Alpaka-based modules running on a device, using

    eventProducts = cms.untracked.vstring("module")

will cause "module" to run, along with automatic copy of its device products to
the host, and will attempt to clone all device and host products.
To clone only the host product, the branch can be specified explicitly with

    eventProducts = cms.untracked.vstring( "HostProductType_module_*_*" )

.

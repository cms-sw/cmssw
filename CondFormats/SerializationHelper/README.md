# Overview of CondFormats/SerializatonHelper

This package provides a mechanism to load plugins which are capable of doing C++ object serialization/deserialization
using the same mechanism (boost serialization) as the conditions database.


## Using the Code
The plugin factory is named `SerializationHelperFactory` which returns instances of `SerializationHelperBase`. The
latter provides methods to serialize an C++ class to a buffer and to deserialize from a buffer to an instance of a C++
class. Polymorphism is supported for serialization/deserialization.

## Registering a Plugin

The normal way to register a plugin is to just register a plugin for the CondDBESSource. The registration macros for
the CondDBESSource contain calls to the macros used to generate the SerializationHelpers. One caveat is for the case
where the same C++ class is begin stored in multiple Records within CondDBESSource. As the CondDBESSource registration
macro includes both the Record name and the C++ class name, but the SerializationHelper system only cares about the
C++ class name, this situation can cause the registration of the same C++ class multiple times which leads to a
compilation error. This problem can be overcome by replacing one of the `REGISTER_PLUGIN` calls with `REGISTER_PLUGIN_NO_SERIAL`. E.g.

change
```cpp
 REGISTER_PLUGIN(FooRecord, Foo);
 REGISTER_PLUGIN(OtherFooRecord, Foo);
```

to
```cpp
 REGISTER_PLUGIN(FooRecord, Foo);
 REGISTER_PLUGIN_NO_SERIAL(OtherFooRecord, Foo);
```

### Handling polymorphism

If the type retrieved from the EventSetup is actually a base class but what is stored is a concrete instance of
one or more inheriting types, then one must explicitly create a template specialization of `cond::serialization::BaseClassInfo<>`.
The template argument is the name of the base class being used by the EventSetup system. The class variable `kAbstract` states
if an instance of a base class has abstract virtual functions (and therefore one can not create an instance of the base class).
The type `inheriting_classes_t` is a compile time list of the inheriting classes which can be stored.

In addition, one must call the `DEFINE_COND_CLASSNAME` macro for the inheriting classes in order to get their names registered
into the system. This is needed as the name of the actual stored type must be written into the file.

#### Example
Say we have the base class `Base` which is what the EventSetup sees and the class has abstract virtual functions.
Say we have two types `InheritA` and `InheritB` which inherit from `Base` and are actually what is to be stored. Then one would define

```cpp
namespace cond::serialization {
  template<>
  struct  BaseClassInfo<Base> {
    constexpr static bool kAbstract = true;
    using inheriting_classes_t = edm::mpl::Vector<InheritA,InheritB>;
  };
}

DEFINE_COND_CLASSNAME(InheritA)
DEFINE_COND_CLASSNAME(InheritB)
```

This code needs to be in the same file as the calls to `REGISTER_PLUGIN`.
#ifndef DataFormats_TrivialSerialisation_interface_MemoryCopyTraits_h
#define DataFormats_TrivialSerialisation_interface_MemoryCopyTraits_h

#include <cassert>
#include <cstddef>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace ngt {

  // This struct should be specialised for each type that can be safely memcpy'ed.
  //
  // The specialisation shall have two static methods
  //
  //   static std::vector<std::span<std::byte>> regions(T& object);
  //   static std::vector<std::span<const std::byte>> regions(T const& object);
  //
  // that return a vector of address, size pairs. A type that supports this
  // interface can be copied by doing a memcpy of all the address, size pairs from a
  // source object to a destination object.
  //
  //
  // A specialisation may implement the method properties(), which returns the
  // properties of an existing object, which can be used to initialize a newly
  // allocated copy of the object via the initialize() method.
  //
  //   using Properties = ...;
  //   static Properties properties(T const& object);
  //
  // If properties() is not implemented, the initialize() method takes a single
  // argument:
  //
  //   static void initialize(T& object);
  //
  // If properties() is implemented, the initialize() method should take as a
  // second parameter a const reference to a Properties object:
  //
  //   static void initialize(T& object, Properties const& args);
  //
  //
  // A specialisation can optionally provide a static method
  //
  //   static void finalize(T& object);
  //
  // If present, it should be called to restore the object invariants after a
  // memcpy operation.
  //

  template <typename T>
  struct MemoryCopyTraits;

  // Checks if the properties method is defined
  template <typename T>
  concept HasTrivialCopyProperties = requires(T const& object) { ngt::MemoryCopyTraits<T>::properties(object); };

  // Get the return type of properties(...), if it exists.
  template <typename T>
    requires ngt::HasTrivialCopyProperties<T>
  using TrivialCopyProperties = decltype(ngt::MemoryCopyTraits<T>::properties(std::declval<T const&>()));

  // Checks if the declaration of initialize(...) is consistent with the presence or absence of properties.
  template <typename T>
  concept HasValidInitialize =
      // does not have properties(...) and initialize(object) takes a single argument, or
      (not ngt::HasTrivialCopyProperties<T> and
       requires(T& object) { ngt::MemoryCopyTraits<T>::initialize(object); }) or
      // does have properties(...) and initialize(object, props) takes two arguments
      (ngt::HasTrivialCopyProperties<T> and requires(T& object, ngt::TrivialCopyProperties<T> props) {
        ngt::MemoryCopyTraits<T>::initialize(object, props);
      });

  // Checks for const and non const memory regions
  template <typename T>
  concept HasRegions = requires(T& object, T const& const_object) {
    ngt::MemoryCopyTraits<T>::regions(object);
    ngt::MemoryCopyTraits<T>::regions(const_object);
  };

  // Checks if there is a valid specialisation of MemoryCopyTraits for a type T
  template <typename T>
  concept HasMemoryCopyTraits =
      // Has memory regions declared and
      (ngt::HasRegions<T>) and
      // has either no initialize(...) or a valid one
      (not requires { &ngt::MemoryCopyTraits<T>::initialize; } or ngt::HasValidInitialize<T>);

  // Checks if finalize(...) is defined
  template <typename T>
  concept HasTrivialCopyFinalize = requires(T& object) { ngt::MemoryCopyTraits<T>::finalize(object); };

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // Specialisations for various types

  // Specialisation for arithmetic types
  template <typename T>
    requires std::is_arithmetic_v<T>
  struct MemoryCopyTraits<T> {
    static std::vector<std::span<std::byte>> regions(T& object) {
      return {{reinterpret_cast<std::byte*>(&object), sizeof(T)}};
    }  // namespace edm

    static std::vector<std::span<const std::byte>> regions(T const& object) {
      return {{reinterpret_cast<std::byte const*>(&object), sizeof(T)}};
    }
  };

  // Specialisation for std::string
  template <>
  struct MemoryCopyTraits<std::string> {
    using Properties = std::string::size_type;

    static Properties properties(std::string const& object) { return object.size(); }
    static void initialize(std::string& object, Properties const& size) { object.resize(size); }

    static std::vector<std::span<std::byte>> regions(std::string& object) {
      return {{reinterpret_cast<std::byte*>(object.data()), object.size() * sizeof(char)}};
    }

    static std::vector<std::span<const std::byte>> regions(std::string const& object) {
      return {{reinterpret_cast<std::byte const*>(object.data()), object.size() * sizeof(char)}};
    }
  };

  // Specialisation for vectors of arithmetic types
  template <typename T>
    requires(std::is_arithmetic_v<T> and not std::is_same_v<T, bool>)
  struct MemoryCopyTraits<std::vector<T>> {
    using Properties = std::vector<T>::size_type;

    static Properties properties(std::vector<T> const& object) { return object.size(); }
    static void initialize(std::vector<T>& object, Properties const& size) { object.resize(size); }

    static std::vector<std::span<std::byte>> regions(std::vector<T>& object) {
      return {{reinterpret_cast<std::byte*>(object.data()), object.size() * sizeof(T)}};
    }

    static std::vector<std::span<const std::byte>> regions(std::vector<T> const& object) {
      return {{reinterpret_cast<std::byte const*>(object.data()), object.size() * sizeof(T)}};
    }
  };

}  // namespace ngt

#endif  // DataFormats_TrivialSerialisation_interface_MemoryCopyTraits_h

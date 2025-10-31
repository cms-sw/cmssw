#ifndef Dataformats_Common_interface_TrivialCopyTraits_h
#define Dataformats_Common_interface_TrivialCopyTraits_h

#include <cassert>
#include <span>
#include <type_traits>
#include <vector>
#include <string>

namespace edm {

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
  struct TrivialCopyTraits;

  // Checks if the properties method is defined
  template <typename T>
  concept HasTrivialCopyProperties = requires(T const& object) { TrivialCopyTraits<T>::properties(object); };

  // Get the return type of properties(...), if it exists.
  template <typename T>
    requires HasTrivialCopyProperties<T>
  using TrivialCopyProperties = decltype(TrivialCopyTraits<T>::properties(std::declval<T const&>()));

  // Checks if the declaration of initialize(...) is consistent with the presence or absence of properties.
  template <typename T>
  concept HasValidInitialize =
      // does not have properties(...) and initialize(object) takes a single argument
      (not HasTrivialCopyProperties<T> && requires(T& object) { TrivialCopyTraits<T>::initialize(object); }) ||
      // or does have properties(...) and initialize(object, props) takes two arguments
      (HasTrivialCopyProperties<T> &&
       requires(T& object, TrivialCopyProperties<T> props) { TrivialCopyTraits<T>::initialize(object, props); });

  // Checks for const and non const memory regions
  template <typename T>
  concept HasRegions = requires(T& object, T const& const_object) {
    TrivialCopyTraits<T>::regions(object);
    TrivialCopyTraits<T>::regions(const_object);
  };

  // Checks if there is a valid specialisation of TrivialCopyTraits for a type T
  template <typename T>
  concept HasTrivialCopyTraits =
      // Has memory regions declared
      (HasRegions<T>) &&
      // and has either no initialize(...) or a valid one
      (not requires { &TrivialCopyTraits<T>::initialize; } || HasValidInitialize<T>);

  // Checks if finalize(...) is defined
  template <typename T>
  concept HasTrivialCopyFinalize = requires(T& object) { edm::TrivialCopyTraits<T>::finalize(object); };

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // Specialisations for various types

  // Specialisation for arithmetic types
  template <typename T>
    requires std::is_arithmetic_v<T>
  struct TrivialCopyTraits<T> {
    static std::vector<std::span<std::byte>> regions(T& object) {
      return {{reinterpret_cast<std::byte*>(&object), sizeof(T)}};
    }

    static std::vector<std::span<const std::byte>> regions(T const& object) {
      return {{reinterpret_cast<std::byte const*>(&object), sizeof(T)}};
    }
  };

  // Specialisation for std::string
  template <>
  struct TrivialCopyTraits<std::string> {
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
  struct TrivialCopyTraits<std::vector<T>> {
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
}  // namespace edm

#endif  // Dataformats_Common_interface_TrivialCopyTraits_h

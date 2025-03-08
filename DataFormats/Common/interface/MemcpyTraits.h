#ifndef DataFormats_Common_interface_MemcpyTraits_h
#define DataFormats_Common_interface_MemcpyTraits_h

#include <cassert>
#include <type_traits>
#include <utility>
#include <vector>

namespace edm {

  // This struct should be specialised for each type that can be safely memcpy'ed.
  //
  // The specialisation shall have two static methods
  //
  //   static std::vector<std::pair<void*, std::size_t>> regions(T& object);
  //   static std::vector<std::pair<void const*, std::size_t>> regions(T const& object);
  //
  // that return a vector of address, size pairs.
  // A type that supports this interface can be copied by doing a memcpy of all
  // the address, size pairs from a source object to a destination object.
  //
  // A specialisation can optionally provide two static methods
  //
  //   static std::vector<std::size_t> properties(T const& object);
  //   static void initialise(T& object, std::vector<std::size_t> const& args);
  //
  // If present, the first may be called to query some type-specific parameters,
  // while the second can use them to allocate the destination object.
  //
  // A specialisation can optionally provide a static method
  //
  //   static void finalize(T& object);
  //
  // If present, it should be called to restore the object invariants after a
  // memcpy operation.

  template <typename T>
  struct MemcpyTraits;

  // Specialisation for arithmetic types
  template <typename T>
    requires std::is_arithmetic_v<T>
  struct MemcpyTraits<T> {
    using value_type = T;

    static std::vector<std::pair<void*, std::size_t>> regions(value_type& object) {
      return {{&object, sizeof(value_type)}};
    }

    static std::vector<std::pair<void const*, std::size_t>> regions(value_type const& object) {
      return {{&object, sizeof(value_type)}};
    }
  };

  // Specialisation for vectors of arithmetic types
  template <typename T>
    requires(std::is_arithmetic_v<T> and not std::is_same_v<T, bool>)
  struct MemcpyTraits<std::vector<T>> {
    using value_type = std::vector<T>;

    static std::vector<std::size_t> properties(value_type const& object) { return object.size(); }

    static void initialize(value_type& object, std::vector<std::size_t> const& args) {
      assert(args.size() == 1);
      object.resize(args[0]);
    }

    static std::vector<std::pair<void*, std::size_t>> regions(value_type& object) {
      return {{object.data(), object.size() * sizeof(T)}};
    }

    static std::vector<std::pair<void const*, std::size_t>> regions(value_type const& object) {
      return {{object.data(), object.size() * sizeof(T)}};
    }
  };

}  // namespace edm

#endif  // DataFormats_Common_interface_MemcpyTraits_h

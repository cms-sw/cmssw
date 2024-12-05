#ifndef DataFormats_Common_interface_Uninitialized_h
#define DataFormats_Common_interface_Uninitialized_h

/* Uninitialized
 *
 * This is an empty struct used as a tag to signal that a constructor will leave an object (partially) uninitialised,
 * with the assumption that it will be overwritten before being used.
 * One expected use case is to replace the default constructor used when deserialising objects from a ROOT file.
 */

namespace edm {

  struct Uninitialized {};

  constexpr inline Uninitialized kUninitialized;

}  // namespace edm

#endif  // DataFormats_Common_interface_Uninitialized_h

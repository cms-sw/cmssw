#ifndef DataFormats_Portable_interface_PortableCollectionCommon_h
#define DataFormats_Portable_interface_PortableCollectionCommon_h

#include <array>
#include <cstddef>
#include <type_traits>

namespace portablecollection {

  // concept to check if a Layout has a static member blocksNumber
  template <class L>
  concept hasBlocksNumber = requires { L::blocksNumber; };

}  // namespace portablecollection

#endif  // DataFormats_Portable_interface_PortableCollectionCommon_h

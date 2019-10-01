#ifndef MagneticField_GeomBuilder_printUniqueNames_h
#define MagneticField_GeomBuilder_printUniqueNames_h

#include "BaseVolumeHandle.h"

namespace magneticfield {

  /// Just for debugging...
  void printUniqueNames(handles::const_iterator begin, handles::const_iterator end, bool uniq = true);
}  // namespace magneticfield

#endif

#ifndef MagneticField_GeomBuilder_InterpolatorBuilder_h
#define MagneticField_GeomBuilder_InterpolatorBuilder_h
// -*- C++ -*-
//
// Package:     MagneticField/GeomBuilder
// Class  :     InterpolatorBuilder
//
/**\class InterpolatorBuilder InterpolatorBuilder.h "InterpolatorBuilder.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Christopher Jones
//         Created:  Tue, 17 May 2022 20:50:20 GMT
//

// system include files
#include <string>
#include <optional>
#include <unordered_map>

// user include files
#include "MagneticField/Interpolation/interface/MagProviderInterpol.h"
#include "MagneticField/Interpolation/interface/binary_ifstream.h"
#include "DD4hep_volumeHandle.h"

// forward declarations

namespace magneticfield {
  class InterpolatorBuilder {
  public:
    InterpolatorBuilder(std::string iTableSet, bool useMergeFileIfAvailable = true);

    InterpolatorBuilder(const InterpolatorBuilder&) = delete;                   // stop default
    const InterpolatorBuilder& operator=(const InterpolatorBuilder&) = delete;  // stop default

    // ---------- const member functions ---------------------

    // ---------- static member functions --------------------

    // ---------- member functions ---------------------------
    std::unique_ptr<MagProviderInterpol> build(volumeHandle const*);

  private:
    // ---------- member data --------------------------------
    std::string tableSet_;
    std::unordered_map<std::string, unsigned int> offsets_;
    std::optional<interpolation::binary_ifstream> stream_;
  };
}  // namespace magneticfield
#endif

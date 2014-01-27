#ifndef CondFormats_Serialization_Archive_H
#define CondFormats_Serialization_Archive_H

// Missing in EOS' portable archive
#include <cassert>

#include "CondFormats/Serialization/interface/eos/portable_iarchive.hpp"
#include "CondFormats/Serialization/interface/eos/portable_oarchive.hpp"

namespace cond {
namespace serialization {

  typedef eos::portable_iarchive InputArchive;
  typedef eos::portable_oarchive OutputArchive;

}
}

#endif

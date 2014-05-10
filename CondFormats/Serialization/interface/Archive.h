#pragma once

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

